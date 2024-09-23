import networkx as nx
import os
from os.path import join

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def print_nx_network_full_info(nx_graph):                                            
    print('====Nodes info====')
    for node, node_data in nx_graph.nodes(data=True):
        print(node, node_data)
    
    # print('====Edges info====')
    # for source_node, target_node, edge_data in nx_graph.edges(data=True):
    #     print(source_node, target_node, edge_data)

# 获取cfg图和cg图中相互有的图节点（即函数节点）
def mapping_cfg_and_cg_node_labels(cfg, call_graph):
    dict_node_label_cfg_and_cg = {}

    for node, node_data in cfg.nodes(data=True):
        if node_data['node_type'] == 'FUNCTION_NAME':
            if node_data['label'] not in dict_node_label_cfg_and_cg:
                dict_node_label_cfg_and_cg[node_data['label']] = None
            # else:
            #     print(node_data['label'], 'is existing.')

            dict_node_label_cfg_and_cg[node_data['label']] = {
                'cfg_node_id': node,
                'cfg_node_type': node_data['node_type']
            }
    
    
    for node, node_data in call_graph.nodes(data=True):
        if node_data['label'] in dict_node_label_cfg_and_cg: # 这里判断cg节点是否在cfg中。若有则将cg节点信息写入到节点中，让节点同时存有cfg和cg信息。
            dict_node_label_cfg_and_cg[node_data['label']]['call_graph_node_id'] = node
            dict_node_label_cfg_and_cg[node_data['label']]['call_graph_node_type'] = node_data['node_type'].upper()
        else:
            print(node_data['label'], ' is not existing.')


    # remove node labels are not existing in the call graph; cg图中是没有变量节点的。
    temp_dict = dict(dict_node_label_cfg_and_cg)
    for key, value in temp_dict.items():
        if 'call_graph_node_id' not in value or 'call_graph_node_type' not in value:
            dict_node_label_cfg_and_cg.pop(key, None)

    return dict_node_label_cfg_and_cg

# 把cg的边添加到cfg中。相当于把cg融合到cfg中。
def add_new_cfg_edges_from_call_graph(cfg, dict_node_label, call_graph):
    list_new_edges_cfg = []
    for source, target, edge_data in call_graph.edges(data=True):
        source_cfg = None
        target_cfg = None
        edge_data_cfg = edge_data
        for value in dict_node_label.values():
            if value['call_graph_node_id'] == source:
                source_cfg = value['cfg_node_id']
            
            if value['call_graph_node_id'] == target:
                target_cfg = value['cfg_node_id']
        
        if source_cfg is not None and target_cfg is not None:
            list_new_edges_cfg.append((source_cfg, target_cfg, edge_data_cfg))
    
    cfg.add_edges_from(list_new_edges_cfg)

    return cfg
    
def update_cfg_node_types_by_call_graph_node_types(cfg, dict_node_label):
    for value in dict_node_label.values():
        cfg_node_id = value['cfg_node_id']
        cfg.nodes[cfg_node_id]['node_type'] = value['call_graph_node_type']


def combin_all_graph():
    ROOT = '../data/ge-sc-data/source_code'
    # bug_type = {'access_control': 57, 'arithmetic': 60, 'denial_of_service': 46,
    #           'front_running': 44, 'reentrancy': 71, 'time_manipulation': 50,
    #           'unchecked_low_level_calls': 95}

    # TODO: chang types!
    bug_type = {'unchecked_low_level_calls': 95}
    for bug, counter in bug_type.items():

        source = f'{ROOT}/{bug}/all_clean'
        smart_contracts = [join(source, f) for f in os.listdir(source) if f.endswith('.sol')]

        for name_item in smart_contracts:
            file_name_sc = name_item.split('/')[-1:][0]
            input_cfg_path = f'{ROOT}/{bug}/all_cfg_graph/{file_name_sc}.gpickle'
            input_call_graph_path = f'{ROOT}/{bug}/all_cg_graph/{file_name_sc}.gpickle'

            input_cfg = nx.read_gpickle(input_cfg_path)
            input_call_graph = nx.read_gpickle(input_call_graph_path)
            dict_node_label_cfg_and_cg = mapping_cfg_and_cg_node_labels(input_cfg, input_call_graph)  # 获取cfg和cg图中共有的节点
            merged_graph = add_new_cfg_edges_from_call_graph(input_cfg, dict_node_label_cfg_and_cg,
                                                             input_call_graph)  # 将cg中的边添加到cfg图中

            # output_path = f'{ROOT}/{bug}/curated/cfg_cg_compressed_graphs.gpickle'
            output_path = f'{ROOT}/{bug}/all_mixed_graph/{file_name_sc}.gpickle'
            update_cfg_node_types_by_call_graph_node_types(merged_graph,
                                                           dict_node_label_cfg_and_cg)  # 将cfg图中的函数节点类型变为cg图中的节点类型。即“FUNCTION_NAME”变为"CONTRACT_FUNCTION"或"FALLBACK_FUNCTION"
            nx.write_gpickle(merged_graph, output_path)
            print(f'{bug}图写入完成;{output_path}')
            # 清除当前的绘图区域
            plt.clf()
            # nx.draw(merge_contract_graph, with_labels=True)
            nx.draw(merged_graph)
            plt.show()
            plt.savefig(f'{ROOT}/{bug}/all_mixed_graph/{file_name_sc}.png')  # 保存图形到文件


def combina_one_graph(file_name_sc):
    ROOT = '../data/ge-sc-data/source_code'
    bug = 'access_control'

    input_cfg_path = f'{ROOT}/{bug}/all_cfg_graph/{file_name_sc}.gpickle'
    input_call_graph_path = f'{ROOT}/{bug}/all_cg_graph/{file_name_sc}.gpickle'

    input_cfg = nx.read_gpickle(input_cfg_path)
    input_call_graph = nx.read_gpickle(input_call_graph_path)

    dict_node_label_cfg_and_cg = mapping_cfg_and_cg_node_labels(input_cfg, input_call_graph)  # 获取cfg和cg图中共有的节点
    merged_graph = add_new_cfg_edges_from_call_graph(input_cfg, dict_node_label_cfg_and_cg,
                                                     input_call_graph)  # 将cg中的边添加到cfg图中

    # output_path = f'{ROOT}/{bug}/curated/cfg_cg_compressed_graphs.gpickle'
    output_path = f'{ROOT}/{bug}/all_mixed_graph/{file_name_sc}.gpickle'
    update_cfg_node_types_by_call_graph_node_types(merged_graph,
                                                   dict_node_label_cfg_and_cg)  # 将cfg图中的函数节点类型变为cg图中的节点类型。即“FUNCTION_NAME”变为"CONTRACT_FUNCTION"或"FALLBACK_FUNCTION"
    nx.write_gpickle(merged_graph, output_path)
    print(f'{bug}图写入完成;{output_path}')

if __name__ == '__main__':
    combin_all_graph()
    # combina_one_graph( f'buggy_14.sol')
    # combina_one_graph( f'simple_suicide.sol')
