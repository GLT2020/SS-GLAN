import os

import re
import logging
import json
# import pygraphviz as pgv
import networkx as nx

from copy import deepcopy
from os.path import join
from scipy.integrate._ivp.radau import C
from slither.slither import Slither
from collections import defaultdict
from networkx.algorithms import cluster
from slither.core.cfg.node import Node, NodeType
from tqdm import tqdm

from slither.printers.call import call_graph
from slither.printers.abstract_printer import AbstractPrinter
from slither.core.declarations.solidity_variables import SolidityFunction
from slither.core.declarations.function import Function
from slither.core.variables.variable import Variable

import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

logger = logging.getLogger("Slither-simil")


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



# return graph edge with edge type
def _edge(from_node, to_node, edge_type, label):
    return (from_node, to_node, edge_type, label)
    

# return unique id for contract function to use as node name
def _function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc):
    node_function_source_code_lines = function.source_mapping['lines']
    vulnerabilities_in_sc = revert_vulnerabilities_in_sc_from_tuple(tuple_vulnerabilities_in_sc)
    node_function_info_vulnerabilities = get_vulnerabilities_of_node_by_source_code_line(node_function_source_code_lines, vulnerabilities_in_sc)

    node_info = {
        'node_id': f"{filename_input}_{contract.id}_{contract.name}_{function.full_name}",
        'label': f"{filename_input}_{contract.name}_{function.full_name}",
        'function_fullname': function.full_name, 
        'contract_name': contract.name, 
        'source_file': filename_input,
        'node_function_info_vulnerabilities': parse_vulnerabilities_in_sc_to_tuple(node_function_info_vulnerabilities),
        'node_source_code_lines': tuple(node_function_source_code_lines)
    }

    # return f"{filename_input}_{contract.id}_{contract.name}_{function.full_name}"
    return node_info

# return unique id for solidity function to use as node name
def _solidity_function_node(solidity_function):
    # node_function_source_code_lines = solidity_function.source_mapping['lines']
    node_info = {
        'node_id': f"[Solidity]_{solidity_function.full_name}",
        'label': f"[Solidity]_{solidity_function.full_name}",
        'function_fullname': solidity_function.full_name,
        'contract_name': None,
        'source_file': None,
        'node_function_info_vulnerabilities': None,
        'node_source_code_lines': None
    }
    # return f"[Solidity]_{solidity_function.full_name}"
    return node_info

# return node info from a node tupple
def _get_node_info(tuple_node):
    if tuple_node[0][0] == 'node_id':
        node_id = tuple_node[0][1]
    if tuple_node[1][0] == 'label':
        node_label = tuple_node[1][1]
    if tuple_node[2][0] == 'function_fullname':
        function_fullname = tuple_node[2][1]
    if tuple_node[3][0] == 'contract_name':
        contract_name = tuple_node[3][1]
    if tuple_node[4][0] == 'source_file':
        source_file = tuple_node[4][1]
    if tuple_node[5][0] == 'node_function_info_vulnerabilities':
        node_function_info_vulnerabilities = revert_vulnerabilities_in_sc_from_tuple(tuple_node[5][1])
    if tuple_node[6][0] == 'node_source_code_lines':
        node_function_source_code_lines = list(tuple_node[6][1])
    
    if len(node_function_info_vulnerabilities) == 0:
        node_function_info_vulnerabilities = None

    if 'fallback' in node_id:
        node_type = 'fallback_function'
    elif '[Solidity]' in node_id:
        node_type = 'fallback_function'
    else:
        node_type = 'contract_function'
    
    return node_id, node_label, node_type, function_fullname, contract_name, source_file, node_function_info_vulnerabilities, node_function_source_code_lines

# return edge info from a contract call tuple
def _add_edge_info_to_nxgraph(contract_call, nx_graph):
    source = contract_call[0]
    source_node_id, source_label, source_type, source_function_fullname, source_contract_name, \
    source_source_file, source_node_function_info_vulnerabilities, source_node_function_source_code_lines = _get_node_info(source)

    if source_node_id not in nx_graph.nodes(): # 如果调用边的源节点不在图中，则添加该节点
        nx_graph.add_node(source_node_id, label=source_label, node_type=source_type,
                          node_info_vulnerabilities=source_node_function_info_vulnerabilities,
                          node_source_code_lines=source_node_function_source_code_lines,
                          function_fullname=source_function_fullname, contract_name=source_contract_name,
                          source_file=source_source_file)

    target = contract_call[1]
    target_node_id, target_label, target_type, target_function_fullname, target_contract_name, \
    target_source_file, target_node_function_info_vulnerabilities, target_node_function_source_code_lines = _get_node_info(target)

    if target_node_id not in nx_graph.nodes(): # 如果调用边的目标节点不在图中，则添加该节点
        nx_graph.add_node(target_node_id, label=target_label, node_type=target_type,
                          node_info_vulnerabilities=target_node_function_info_vulnerabilities,
                          node_source_code_lines=target_node_function_source_code_lines,
                          function_fullname=target_function_fullname, contract_name=target_contract_name,
                          source_file=target_source_file)

    edge_type = contract_call[2]
    edge_label = contract_call[3]

    nx_graph.add_edge(source_node_id, target_node_id, label=edge_label, edge_type=edge_type)

# pylint: disable=too-many-arguments
def _process_internal_call(
    contract,
    function,
    internal_call,
    contract_calls,
    solidity_functions,
    solidity_calls,
    filename_input,
    vulnerabilities_in_sc=[]
):
    tuple_vulnerabilities_in_sc = parse_vulnerabilities_in_sc_to_tuple(vulnerabilities_in_sc)
    if isinstance(internal_call, (Function)): # 判断内部调用的是否为合约文件中的函数
        # print('tuple:', tuple(_function_node(contract, function, filename_input).items()))
        contract_calls[contract].add(
            _edge(
                tuple(_function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc).items()),
                tuple(_function_node(contract, internal_call, filename_input, tuple_vulnerabilities_in_sc).items()),
                edge_type='internal_call',
                label='internal_call'
            )
        )

    elif isinstance(internal_call, (SolidityFunction)): # 判断内部调用的是否为语言的内部函数
        solidity_functions.add(tuple(_solidity_function_node(internal_call).items()))
        solidity_calls.add(
            _edge(
                tuple(_function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc).items()),
                tuple(_solidity_function_node(internal_call).items()),
                edge_type='solidity_call',
                label='solidity_call'
            )
        )

def _process_external_call(
    contract,
    function,
    external_call,
    contract_functions,
    external_calls,
    all_contracts,
    filename_input,
    vulnerabilities_in_sc=[]
):
    tuple_vulnerabilities_in_sc = parse_vulnerabilities_in_sc_to_tuple(vulnerabilities_in_sc)
    external_contract, external_function = external_call
    
    if not external_contract in all_contracts:
        return

    # add variable as node to respective contract
    if isinstance(external_function, (Variable)):
        contract_functions[external_contract].add(tuple(
                _function_node(external_contract, external_function, filename_input, tuple_vulnerabilities_in_sc).items()))


    external_calls.add(
        _edge(
            tuple(_function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc).items()),
            tuple(_function_node(external_contract, external_function, filename_input, tuple_vulnerabilities_in_sc).items()),
            edge_type='external_call',
            label='external_call'
        )
    )

def _render_internal_calls(nx_graph, contract, contract_functions, contract_calls):
    if len(contract_functions[contract]) > 0:
        for contract_function in contract_functions[contract]:     # 将函数转为nx的图节点结构
            node_id, node_label, node_type, function_fullname, contract_name, source_file, \
            node_function_info_vulnerabilities, node_function_source_code_lines = _get_node_info(contract_function)

            nx_graph.add_node(node_id, label=node_label, node_type=node_type,
                              node_info_vulnerabilities=node_function_info_vulnerabilities,
                              node_source_code_lines=node_function_source_code_lines,
                              function_fullname=function_fullname, contract_name=contract_name,
                              source_file=source_file)
    
    if len(contract_calls[contract]) > 0:  # 将内部调用边转为nx的边结构
        for contract_call in contract_calls[contract]:
            _add_edge_info_to_nxgraph(contract_call, nx_graph)


def _render_solidity_calls(nx_graph, solidity_functions, solidity_calls):
    if len(solidity_functions) > 0:
        for solidity_function in solidity_functions:
            # print(solidity_function)
            node_id, node_label, node_type, function_fullname, contract_name, source_file, \
            node_function_info_vulnerabilities, node_function_source_code_lines = _get_node_info(solidity_function)

            nx_graph.add_node(node_id, label=node_label, node_type=node_type,
                              node_info_vulnerabilities=node_function_info_vulnerabilities,
                              node_source_code_lines=node_function_source_code_lines,
                              function_fullname=function_fullname, contract_name=contract_name,
                              source_file=source_file)
    
    if len(solidity_calls) > 0:
        for solidity_call in solidity_calls:
            _add_edge_info_to_nxgraph(solidity_call, nx_graph)

def _render_external_calls(nx_graph, external_calls):
    if len(external_calls) > 0:
        for external_call in external_calls:
            _add_edge_info_to_nxgraph(external_call, nx_graph)

def _process_function(
    contract,
    function,
    contract_functions,
    contract_calls,
    solidity_functions,
    solidity_calls,
    external_calls,
    all_contracts,
    filename_input,
    vulnerabilities_in_sc=[]
):  
    tuple_vulnerabilities_in_sc = parse_vulnerabilities_in_sc_to_tuple(vulnerabilities_in_sc)
    contract_functions[contract].add(tuple(
        _function_node(contract, function, filename_input, tuple_vulnerabilities_in_sc).items())
    )

    for internal_call in function.internal_calls:
        _process_internal_call(
            contract,
            function,
            internal_call,
            contract_calls,
            solidity_functions,
            solidity_calls,
            filename_input,
            vulnerabilities_in_sc
        )

    for external_call in function.high_level_calls:
        _process_external_call(
            contract,
            function,
            external_call,
            contract_functions,
            external_calls,
            all_contracts,
            filename_input,
            vulnerabilities_in_sc
        )

def _process_functions(functions, filename_input, vulnerabilities_in_sc=None):
    contract_functions = defaultdict(set)  # contract -> contract functions nodes 存储合约的函数，每个函数作为一个节点
    contract_calls = defaultdict(set)  # contract -> contract calls edges 存储合约节点调用的边。分为内部、外部调用边

    solidity_functions = set()  # solidity function nodes
    solidity_calls = set()  # solidity calls edges
    external_calls = set()  # external calls edges

    all_contracts = set() # 获取合约文件中的合约

    for function in functions:
        all_contracts.add(function.contract_declarer) # function.contract_declarer 表示是哪个合约声明的这个函数

    for function in functions:
        _process_function(
            function.contract_declarer,
            function,
            contract_functions,
            contract_calls,
            solidity_functions,
            solidity_calls,
            external_calls,
            all_contracts,
            filename_input,
            vulnerabilities_in_sc
        )

    # print('contract_functions:', contract_functions)
    # print('solidity_functions:', solidity_functions)
    # print('contract_calls:', contract_calls)
    # print('solidity_calls:', solidity_calls)
    # print('external_calls:', external_calls)
    # print('all_contracts:', all_contracts)

    all_contracts_graph = nx.MultiDiGraph()
    for contract in all_contracts:
        _render_internal_calls(all_contracts_graph, contract,
                               contract_functions, contract_calls)
    
    # _render_solidity_calls(all_contracts_graph, solidity_functions, solidity_calls)
    _render_external_calls(all_contracts_graph, external_calls)

    return all_contracts_graph
# 将漏洞信息转为元组形式
def parse_vulnerabilities_in_sc_to_tuple(vulnerabilities_in_sc):
    vul_info = list()

    if vulnerabilities_in_sc is not None:
        for vul in vulnerabilities_in_sc:
            for key, value in vul.items():
                if key == 'lines':
                    vul[key] = tuple(value)
        
        for vul in vulnerabilities_in_sc:
            vul_info.append(tuple(vul.items()))

    vul_info = tuple(vul_info)
    return vul_info
# 将漏洞数据从元组转为字典列表形式
def revert_vulnerabilities_in_sc_from_tuple(tuple_vulnerabilities_in_sc):
    vulnerabilities_in_sc = list(tuple_vulnerabilities_in_sc)

    vul_info = []
    if len(vulnerabilities_in_sc) > 0:
        for vul in vulnerabilities_in_sc:
            dct = dict((x, y) for x, y in vul)
            for key, val in dct.items():
                if key == 'lines':
                    dct[key] = list(val)

            vul_info.append(dct)

    return vul_info


def get_vulnerabilities(file_name_sc, vulnerabilities):
    list_vulnerability_in_sc = None
    if vulnerabilities is not None:
        for vul_item in vulnerabilities:
            if file_name_sc == vul_item['name']:
                list_vulnerability_in_sc = vul_item['vulnerabilities']
            
    return list_vulnerability_in_sc
# 根据漏洞信息，返回该函数节点是否有漏洞
def get_vulnerabilities_of_node_by_source_code_line(source_code_lines, list_vul_info_sc):
    if list_vul_info_sc is not None:
        list_vulnerability = []
        for vul_info_sc in list_vul_info_sc:
            vulnerabilities_lines = vul_info_sc['lines']
            # for source_code_line in source_code_lines:
            #     for vulnerabilities_line in vulnerabilities_lines:
            #         if source_code_line == vulnerabilities_line:
            #             list_vulnerability.append(vul_info_sc)
            interset_lines = set(vulnerabilities_lines).intersection(set(source_code_lines))
            if len(interset_lines) > 0:
                list_vulnerability.append(vul_info_sc)

    else:
        list_vulnerability = None
    
    if list_vulnerability is None or len(list_vulnerability) == 0:
        node_info_vulnerabilities = None
    else:
        node_info_vulnerabilities = list_vulnerability

    return node_info_vulnerabilities


def extract_graph(source_path, output, vulnerabilities=None):
    sc_version = get_solc_version(source_path)
    solc_compiler = f'.solc-select/artifacts/solc-{sc_version}'
    if not os.path.exists(solc_compiler):
        solc_compiler = f'.solc-select/artifacts/solc-0.4.25'
    file_name_sc = source_path.split('/')[-1]
    try:
        slither = Slither(source_path, solc=solc_compiler)
    except Exception as e:
        return 0

    vulnerabilities_info_in_sc = get_vulnerabilities(file_name_sc, vulnerabilities)
    call_graph_printer = GESCPrinters(slither, file_name_sc, logger, vulnerabilities_info_in_sc)
    all_contracts_call_graph = call_graph_printer.generate_all_contracts_call_graph()  
    nx.write_gpickle(all_contracts_call_graph, join(output, file_name_sc))
    return 1


def compress_full_smart_contracts(smart_contracts, output, vulnerabilities=None):
    full_graph = None
    count = 0
    for sc in tqdm(smart_contracts):
        print(sc)
        sc_version = get_solc_version(sc)
        # solc_compiler = f'.solc-select/artifacts/solc-{sc_version}'
        # if not os.path.exists(solc_compiler):
        #     solc_compiler = f'.solc-select/artifacts/solc-0.4.25'
        file_name_sc = sc.split('/')[-1:][0]
        # try:
        #     slither = Slither(sc, solc=solc_compiler)
        #     count += 1
        # except Exception as e:
        #     print(e)
        #     continue

        # 使用solc-select时使用。由于过低的版本solc-select 安装不了
        try:
            solc_path = set_solc_version(sc_version)
        except Exception as e:
            solc_path = set_solc_version('0.4.25')

        try:
            # slither = Slither(sc, solc=solc_compiler)

            slither = Slither(sc, solc=solc_path)

            count += 1
        except Exception as e:
            print('exception ', e)
            continue

        vulnerabilities_info_in_sc = get_vulnerabilities(file_name_sc, vulnerabilities)

        print(file_name_sc, vulnerabilities_info_in_sc)

        call_graph_printer = GESCPrinters(slither, file_name_sc, logger, vulnerabilities_info_in_sc)
        # print(call_graph_printer.filename)

        all_contracts_call_graph = call_graph_printer.generate_all_contracts_call_graph()
        # call_graph_printer.output(file_name_sc + 'GESC')

        nx.write_gpickle(all_contracts_call_graph, output + f'{file_name_sc}.gpickle')

        # 清除当前的绘图区域
        plt.clf()
        # nx.draw(merge_contract_graph, with_labels=True)
        nx.draw(all_contracts_call_graph)
        plt.show()
        plt.savefig(output + f'{file_name_sc}.png')  # 保存图形到文件

        # if full_graph is None:
        #     full_graph = deepcopy(all_contracts_call_graph)
        # elif all_contracts_call_graph is not None:
        #     full_graph = nx.disjoint_union(full_graph, all_contracts_call_graph)
    
    # print(f'{count}/{len(smart_contracts)}')
    # print(nx.info(full_graph))
    # print('Full graph nodes:', full_graph.nodes(data=True))
    # for node, node_data in full_graph.nodes(data=True):
    #     if node_data['node_info_vulnerabilities'] is not None:
    #         print(node, node_data)
    # print('=======================================================')
    # print('Full graph edges:', full_graph.edges(data=True))

    # nx.nx_agraph.write_dot(full_graph, join(output, 'compress_call_graphs.dot'))
    # print('Dumped succesfully:', join(output, 'compress_call_graphs.dot'))
    # nx.write_gpickle(full_graph, join(output, 'compress_call_graphs.gpickle'))
    # print('Dumped succesfully:', join(output, 'compress_call_graphs.gpickle'))
    # nx.nx_agraph.write_dot(full_graph, join(output, 'compress_call_graphs_no_solidity_calls_buggy.dot'))
    # print('Dumped succesfully:', join(output, 'compress_call_graphs_no_solidity_calls_buggy.dot'))

    # nx.write_gpickle(full_graph, output)
    # print('Dumped succesfully:', output)

def merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files):
    result = list()
    for f1 in list_vulnerabilities_json_files:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))

    return result


class GESCPrinters(AbstractPrinter):
    ARGUMENT = 'call-graph'
    HELP = 'Export the call-graph of the contracts to a dot file and a gpickle file'

    WIKI = 'https://github.com/trailofbits/slither/wiki/Printer-documentation#call-graph'

    def __init__(self, slither, filename, logger, vulnerabilities_in_sc=None):
        super().__init__(slither, logger)
        self.filename = filename
        self.vulnerabilities_in_sc = vulnerabilities_in_sc

    def generate_all_contracts_call_graph(self):
        # Avoid dupplicate funcitons due to different compilation unit 这里获取了合约文件中所有的函数
        all_functionss = [
            compilation_unit.functions for compilation_unit in self.slither.compilation_units
        ]
        all_functions = [item for sublist in all_functionss for item in sublist] # 展开列表中的函数对象
        all_functions_as_dict = { # 生成一个函数名为key，函数对象为vul的字典
            function.canonical_name: function for function in all_functions
        }

        all_contracts_call_graph = _process_functions(all_functions_as_dict.values(), self.filename, self.vulnerabilities_in_sc)

        return all_contracts_call_graph

    def output(self, filename):
        """
        Output the graph in filename
        Args:
            filename(string)
        """

if __name__ == '__main__':
    # smart_contract_path = 'data/extracted_source_code/' 
    # output_path = 'data/extracted_source_code/'

    ROOT = '../data/ge-sc-data/source_code'
    # bug_type = {'access_control': 57*2, 'arithmetic': 60*2, 'denial_of_service': 46*2,
    #           'front_running': 44*2, 'reentrancy': 71*2, 'time_manipulation': 50*2,
    #           'unchecked_low_level_calls': 95*2}

    # TODO: chang types!
    bug_type = {'unchecked_low_level_calls': 95*2}
    for bug, counter in bug_type.items():
        source = f'{ROOT}/{bug}/all_clean'

        output = f'{ROOT}/{bug}/all_cg_graph/'
        smart_contracts = [join(source, f) for f in os.listdir(source) if f.endswith('.sol')]
        list_vulnerabilities_json_files = [
            f'{ROOT}/{bug}/preprocess_all.json',
            # f'{ROOT}/{bug}/vulnerabilities.json',
        ]
        data_vulnerabilities = merge_data_from_vulnerabilities_json_files(list_vulnerabilities_json_files)
        compress_full_smart_contracts(smart_contracts, output, vulnerabilities=data_vulnerabilities)
        

