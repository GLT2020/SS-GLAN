import os
import json

from os.path import join
from shutil import copy
from copy import deepcopy
from re import L
from typing import Pattern
from tqdm import tqdm
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import networkx as nx
from slither.slither import Slither
from slither.core.cfg.node import NodeType
from solc import install_solc

import subprocess


pattern =  re.compile(r'\d.\d.\d+')
def get_solc_version(source):
    with open(source, 'r') as f:
        line = f.readline()
        while line:
            if 'pragma solidity' in line:
                if len(pattern.findall(line)) > 0:
                    return pattern.findall(line)[0]
                else:
                    return '0.4.25'
            line = f.readline()
    return '0.4.25'

# 获取节点的信息
def get_node_info(node, list_vulnerabilities_info_in_sc):
    node_label = "Node Type: {}\n".format(str(node.type))
    node_type = str(node.type)
    if node.expression:
        node_label += "\nEXPRESSION:\n{}\n".format(node.expression)
        node_expression = str(node.expression)
    else:
        node_expression = None
    if node.irs:
        node_label += "\nIRs:\n" + "\n".join([str(ir) for ir in node.irs])
        node_irs = "\n".join([str(ir) for ir in node.irs])
    else:
        node_irs = None
    # 获取该节点对应的行数
    node_source_code_lines = node.source_mapping['lines']
    node_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_source_code_lines, list_vulnerabilities_info_in_sc)
    
    return node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines

def get_vulnerabilities(file_name_sc, vulnerabilities):
    list_vulnerability_in_sc = None
    if vulnerabilities is not None:
        for vul_item in vulnerabilities:
            if file_name_sc == vul_item['name']:
                list_vulnerability_in_sc = vul_item['vulnerabilities']
            
    return list_vulnerability_in_sc

# 根据节点行和漏洞行，判断该节点是否存在漏洞
def get_vulnerabilities_of_node_by_source_code_line(source_code_lines, list_vul_info_sc):
    if list_vul_info_sc is not None:
        list_vulnerability = []
        for vul_info_sc in list_vul_info_sc:
            vulnerabilities_lines = vul_info_sc['lines']
            # for source_code_line in source_code_lines:
            #     for vulnerabilities_line in vulnerabilities_lines:
            #         if source_code_line == vulnerabilities_line:
            #             list_vulnerability.append(vul_info_sc)
            interset_lines = set(vulnerabilities_lines).intersection(set(source_code_lines)) # 返回节点行与漏洞行的交集。
            if len(interset_lines) > 0:
                list_vulnerability.append(vul_info_sc)

    else:
        list_vulnerability = None
    
    if list_vulnerability is None or len(list_vulnerability) == 0:
        node_info_vulnerabilities = None
    else:
        node_info_vulnerabilities = list_vulnerability

    return node_info_vulnerabilities

# 自己写的，用于自动跟换solc版本。存在的问题：没有判断是否已经下载过这个版本了，所以会反复下载
def set_solc_version(version):
    # 获取当前环境变量
    env = os.environ.copy()
    # 获取conda虚拟环境的路径
    conda_env_path = os.path.dirname(os.path.dirname(os.__file__))

    # 这里获取的虚拟环境路径回到/lib下，正常应该在上一级文件。使用 split() 方法分割路径字符串，并取前面的部分
    parts = conda_env_path.split("/")
    # 将除最后一个元素外的所有元素连接起来
    new_path = "/".join(parts[:-1])
    new_path = os.path.join(new_path, 'bin/')
    solc_path = os.path.join(new_path, 'solc')

    # 添加conda虚拟环境的bin目录到PATH
    env['PATH'] = os.pathsep.join([env.get('PATH', ''), new_path])

    # 先尝试直接use某个版本，如果没有再进行安装use
    try:
        subprocess.run(['solc-select', 'use', version], check=True, env=env)
    except Exception as e:
        subprocess.run(['solc-select', 'install', version], check=True, env=env)
        subprocess.run(['solc-select', 'use', version], check=True, env=env)

    # 安装指定版本的solc
    # subprocess.run(['solc-select', 'install', version], check=True, env=env)
    # subprocess.run(['solc-select install ' + version], check=True, env=env, shell=True)

    # 使用指定版本的solc
    # subprocess.run(['solc-select', 'use', version], check=True, env=env)
    # subprocess.run(['solc-select use ' + version], check=True, env=env, shell=True)

    return solc_path


def compress_full_smart_contracts(smart_contracts, input_graph, output, vulnerabilities=None):
    full_graph = None
    if input_graph is not None:
        full_graph = nx.read_gpickle(input_graph)
    count = 0
    # 遍历合约列表，利用Slither工具生成CFG图，并存储。
    for sc in tqdm(smart_contracts):
        sc_version = get_solc_version(sc)
        print(f'{sc} - {sc_version}')
        file_name_sc = sc.split('/')[-1:][0]
        bug_type = sc.split('/')[-3]

        # 使用solc-select时使用。由于过低的版本solc-select 安装不了
        try:
            solc_path = set_solc_version(sc_version)
        except Exception as e:
            solc_path = set_solc_version('0.4.25')

        try:
            slither = Slither(sc, solc=solc_path)
            count += 1
        except Exception as e:
            print('exception ', e)
            continue
        # 根据漏洞数据json，获取对应合约的漏洞标签行
        list_vul_info_sc = get_vulnerabilities(file_name_sc, vulnerabilities)

        print(file_name_sc, list_vul_info_sc)

        merge_contract_graph = None

        # 遍历Slither编译后的合约图内容。一个合约文件有多个合约
        for contract in slither.contracts:
            node_count = 0
            merged_graph = None
            for idx, function in enumerate(contract.functions + contract.modifiers):  

                nx_g = nx.MultiDiGraph()
                for nidx, node in enumerate(function.nodes):  # 遍历一个函数的所有节点
                    node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(node, list_vul_info_sc)
                    
                    nx_g.add_node(node.node_id, label=node_label,
                                  node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                  node_info_vulnerabilities=node_info_vulnerabilities,
                                  node_source_code_lines=node_source_code_lines,
                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)

                    node_count += 1


                    if node.type in [NodeType.IF, NodeType.IFLOOP]:
                        true_node = node.son_true
                        if true_node:
                            if true_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(true_node, list_vul_info_sc)
                                nx_g.add_node(true_node.node_id, label=node_label,
                                              node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                              node_info_vulnerabilities=node_info_vulnerabilities,
                                              node_source_code_lines=node_source_code_lines,
                                              function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                                node_count += 1
                            nx_g.add_edge(node.node_id, true_node.node_id, edge_type='if_true', label='True')
                        
                        
                        false_node = node.son_false
                        if false_node:
                            if false_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(false_node, list_vul_info_sc)
                                nx_g.add_node(false_node.node_id, label=node_label,
                                              node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                              node_info_vulnerabilities=node_info_vulnerabilities,
                                              node_source_code_lines=node_source_code_lines,
                                              function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                                node_count += 1
                            nx_g.add_edge(node.node_id, false_node.node_id, edge_type='if_false', label='False')
                            
                    else:
                        for son_node in node.sons: # 遍历该节点的子节点
                            if son_node:
                                if son_node.node_id not in nx_g.nodes(): # 如果子节点不在图中，则添加进去
                                    node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(son_node, list_vul_info_sc)
                                    nx_g.add_node(son_node.node_id, label=node_label,
                                                  node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                                  node_info_vulnerabilities=node_info_vulnerabilities,
                                                  node_source_code_lines=node_source_code_lines,
                                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                                    node_count += 1
                                nx_g.add_edge(node.node_id, son_node.node_id, edge_type='next', label='Next')

                nx_graph = nx_g
                # add FUNCTION_NAME node （增加函数名节点——用于与CG连接的节点。其实在融合时，这个节点将代表cg中的节点）
                node_function_name = file_name_sc + '_' + contract.name + '_' + function.full_name
                node_function_source_code_lines = function.source_mapping['lines']
                node_function_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_function_source_code_lines, list_vul_info_sc)
                nx_graph.add_node(node_function_name, label=node_function_name,
                                  node_type='FUNCTION_NAME', node_expression=None, node_irs=None,
                                  node_info_vulnerabilities=node_function_info_vulnerabilities,
                                  node_source_code_lines=node_function_source_code_lines,
                                  function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                node_count += 1

                if 0 in nx_graph.nodes(): # 将入口节点与新增的函数名节点连接
                    nx_graph.add_edge(node_function_name, 0, edge_type='next', label='Next')
                # 对原来的图中节点进行重新标记。lambda x: 这里会将所有节点遍历，修改节点的名字
                nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.full_name + '_' + str(x), copy=False)

                if merged_graph is None:
                    merged_graph = deepcopy(nx_graph)
                else:# 将之前的图 与 其他函数、修改器图（同一个合约中的不同函数） 合并成一个大图
                    merged_graph = nx.disjoint_union(merged_graph, nx_graph)

            if merge_contract_graph is None: # 将整个合约文件中所有合约的图进行合并
                merge_contract_graph = deepcopy(merged_graph)
            elif merged_graph is not None:
                merge_contract_graph = nx.disjoint_union(merge_contract_graph, merged_graph)

        # 完成一个图的构建，写入一个文件
        print(f'{node_count}/{file_name_sc}')
        # nx.write_gpickle(merge_contract_graph, output + f'{file_name_sc}.gpickle')

        # # 清除当前的绘图区域
        plt.clf()
        # nx.draw(merge_contract_graph, with_labels=True)
        nx.draw(merge_contract_graph)
        plt.show()
        plt.savefig(output + f'{file_name_sc}.png')  # 保存图形到文件

        # 目标是一个合约一个图文件，所以不需要合并全部的图.也就是不需要使用下面的代码
        # if full_graph is None: # 这里是将所有 合约文件的图 合并起来
        #     full_graph = deepcopy(merge_contract_graph)
        # elif merge_contract_graph is not None: # merge_contract_graph 是当前合约图结构
        #     full_graph = nx.disjoint_union(full_graph, merge_contract_graph)

    print(f'{count}/{len(smart_contracts)}')
    # nx.nx_agraph.write_dot(full_graph, output.replace('.gpickle', '.dot'))
    # nx.write_gpickle(full_graph, output+'.gpickle')


def merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files):
    result = list()
    for f1 in list_vulnerabilities_json_files:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    return result

def check_extract_graph(source_path):
    sc_version = get_solc_version(source_path)
    solc_compiler = f'~/.solc-select/artifacts/solc-{sc_version}'
    if not os.path.exists(solc_compiler):
        solc_compiler = f'~/.solc-select/artifacts/solc-0.4.25'
    try:
        slither = Slither(source_path, solc=solc_compiler)
        return 1
    except Exception as e:
        return 0


def extract_graph(source_path, output, vulnerabilities=None):
    sc_version = get_solc_version(source_path)
    solc_compiler = f'~/.solc-select/artifacts/solc-{sc_version}'
    if not os.path.exists(solc_compiler):
        solc_compiler = f'~/.solc-select/artifacts/solc-0.4.25'
    file_name_sc = source_path.split('/')[-1]
    try:
        slither = Slither(source_path, solc=solc_compiler)
    except Exception as e:
        print('exception ', e)
        return 0

    list_vul_info_sc = get_vulnerabilities(file_name_sc, vulnerabilities)

    merge_contract_graph = None
    for contract in slither.contracts:
        merged_graph = None
        for idx, function in enumerate(contract.functions + contract.modifiers):  

            nx_g = nx.MultiDiGraph()
            for nidx, node in enumerate(function.nodes):             
                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(node, list_vul_info_sc)
                
                nx_g.add_node(node.node_id, label=node_label,
                                node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                node_info_vulnerabilities=node_info_vulnerabilities,
                                node_source_code_lines=node_source_code_lines,
                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                
                if node.type in [NodeType.IF, NodeType.IFLOOP]:
                    true_node = node.son_true
                    if true_node:
                        if true_node.node_id not in nx_g.nodes():
                            node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(true_node, list_vul_info_sc)
                            nx_g.add_node(true_node.node_id, label=node_label,
                                            node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                            node_info_vulnerabilities=node_info_vulnerabilities,
                                            node_source_code_lines=node_source_code_lines,
                                            function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                        nx_g.add_edge(node.node_id, true_node.node_id, edge_type='if_true', label='True')
                    
                    
                    false_node = node.son_false
                    if false_node:
                        if false_node.node_id not in nx_g.nodes():
                            node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(false_node, list_vul_info_sc)
                            nx_g.add_node(false_node.node_id, label=node_label,
                                            node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                            node_info_vulnerabilities=node_info_vulnerabilities,
                                            node_source_code_lines=node_source_code_lines,
                                            function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                        nx_g.add_edge(node.node_id, false_node.node_id, edge_type='if_false', label='False')
                        
                else:
                    for son_node in node.sons:
                        if son_node:
                            if son_node.node_id not in nx_g.nodes():
                                node_label, node_type, node_expression, node_irs, node_info_vulnerabilities, node_source_code_lines = get_node_info(son_node, list_vul_info_sc)
                                nx_g.add_node(son_node.node_id, label=node_label,
                                                node_type=node_type, node_expression=node_expression, node_irs=node_irs,
                                                node_info_vulnerabilities=node_info_vulnerabilities,
                                                node_source_code_lines=node_source_code_lines,
                                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
                            nx_g.add_edge(node.node_id, son_node.node_id, edge_type='next', label='Next')

            nx_graph = nx_g
            # add FUNCTION_NAME node
            node_function_name = file_name_sc + '_' + contract.name + '_' + function.full_name
            node_function_source_code_lines = function.source_mapping['lines']
            node_function_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_function_source_code_lines, list_vul_info_sc)
            nx_graph.add_node(node_function_name, label=node_function_name,
                                node_type='FUNCTION_NAME', node_expression=None, node_irs=None,
                                node_info_vulnerabilities=node_function_info_vulnerabilities,
                                node_source_code_lines=node_function_source_code_lines,
                                function_fullname=function.full_name, contract_name=contract.name, source_file=file_name_sc)
            
            if 0 in nx_graph.nodes():
                nx_graph.add_edge(node_function_name, 0, edge_type='next', label='Next')

            nx_graph = nx.relabel_nodes(nx_graph, lambda x: contract.name + '_' + function.full_name + '_' + str(x), copy=False)

            if merged_graph is None:
                merged_graph = deepcopy(nx_graph)
            else:
                merged_graph = nx.disjoint_union(merged_graph, nx_graph)

        if merge_contract_graph is None:
            merge_contract_graph = deepcopy(merged_graph)
        elif merged_graph is not None:
            merge_contract_graph = nx.disjoint_union(merge_contract_graph, merged_graph)
    
    nx.write_gpickle(merge_contract_graph, join(output, file_name_sc))
    return 1

if __name__ == '__main__':

    # smart_contract_path = 'data/clean_71_buggy_curated_0'
    # input_graph = None
    # output_path = 'data/clean_71_buggy_curated_0/cfg_compress_graphs.gpickle'
    # smart_contracts = [join(smart_contract_path, f) for f in os.listdir(smart_contract_path) if f.endswith('.sol')]

    # data_vulnerabilities = None
    # list_vulnerabilities_json_files = [
    #     'data/solidifi_buggy_contracts/reentrancy/vulnerabilities.json',
    #     # 'data/solidifi_buggy_contracts/access_control/vulnerabilities.json',
    #     'data/smartbug-dataset/vulnerabilities.json']
    
    # data_vulnerabilities = merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files)
    
    # compress_full_smart_contracts(smart_contracts, input_graph, output_path, vulnerabilities=data_vulnerabilities)

    ROOT = '../data/ge-sc-data/source_code'
    # bug_type = {'access_control': 57*2, 'arithmetic': 60*2, 'denial_of_service': 46*2,
    #           'front_running': 44*2, 'reentrancy': 71*2, 'time_manipulation': 50*2,
    #           'unchecked_low_level_calls': 95*2}

    # TODO: chang types!
    bug_type = {'reentrancy': 71*2}
    for bug, counter in bug_type.items():
        source = f'{ROOT}/{bug}/all_clean'

        output = f'{ROOT}/{bug}/all_cfg_graph/'

        smart_contracts = [join(source, f) for f in os.listdir(source) if f.endswith('.sol')]
        data_vulnerabilities = None
        list_vulnerabilities_json_files = [
            f'{ROOT}/{bug}/preprocess_all.json',
            # f'{ROOT}/{bug}/vulnerabilities.json',
                                           ]
        data_vulnerabilities = merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files)
        compress_full_smart_contracts(smart_contracts, None, output, vulnerabilities=data_vulnerabilities)