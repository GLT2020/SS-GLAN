import json
import os
import re

# access_control,arithmetic 已经处理好了
# TODO: change type!
type = {
    # 'access_control': 57,
    #     'arithmetic': 60,
    #     'denial_of_service': 46,
    #     'front_running': 44,
    #     'reentrancy': 71,
    #     'time_manipulation': 50,
        'unchecked_low_level_calls': 95
}


def read_vulnerabilities_json_files(list_vulnerabilities_json_files):
    result = list()
    for f1 in list_vulnerabilities_json_files:
        with open(f1, 'r') as infile:
            result.extend(json.load(infile))
    return result


def write_vulnerabilities_json_files(list, path):
    with open(path, 'w') as write_f:
        write_f.write(json.dumps(list, ensure_ascii=False))
    print("写入完成...")


def get_buggy_curated_json():
    for type_key, type_vul in type.items():
        list_vulnerabilities_json_files = [
            './solidifi_buggy_contracts/' + type_key + '/vulnerabilities.json',
            './smartbug-dataset/vulnerabilities.json'
        ]

        data_vulnerabilities = read_vulnerabilities_json_files(list_vulnerabilities_json_files)

        list_buggy_curated_json = list()
        filePath = './ge-sc-data/source_code/' + type_key + '/buggy_curated'
        list_sol_files = os.listdir(filePath)
        count = 0
        for sol_item in list_sol_files:
            for vul_item in data_vulnerabilities:
                if sol_item == vul_item['name']:
                    list_buggy_curated_json.append(vul_item)
                    count += 1

        print(type_key, "count:", type_vul)
        print("count:", count)

        # if count == type_vul:
        write_path = './ge-sc-data/source_code/' + type_key + '/buggy_curated.json'
        write_vulnerabilities_json_files(list_buggy_curated_json, write_path)


# 有的json对不上数，需要找出来是哪个合约
def check_buggy_curated_json(type_key: str):
    list_vulnerabilities_json_files = [
        # './solidifi_buggy_contracts/' + type_key + '/vulnerabilities.json',
        # './smartbug-dataset/vulnerabilities.json'
        './ge-sc-data/source_code/' + type_key + '/buggy_curated.json'
    ]

    data_vulnerabilities = read_vulnerabilities_json_files(list_vulnerabilities_json_files)
    filePath = './ge-sc-data/source_code/' + type_key + '/buggy_curated'
    list_sol_files = os.listdir(filePath)
    print("文件个数：", len(list_sol_files))
    count = 0

    data_vul_name = list()

    for vul_item in data_vulnerabilities:
        data_vul_name.append(vul_item['name'])

    print("json文件个数：", len(data_vul_name))

    # 查看是否是有多的
    for sol_item in list_sol_files:
        if sol_item not in data_vul_name:
            print("多出的合约名：", sol_item)
            count += 1
    print("多出的合约文件数：", count)

    # 查看是否有重复的
    mul_name = list(set([x for x in data_vul_name if data_vul_name.count(x) > 1]))
    print("json中重复的文件：", mul_name)


def get_all_json():
    for type_key, type_vul in type.items():
        list_vulnerabilities_json_files = [
            # './solidifi_buggy_contracts/' + type_key + '/vulnerabilities.json',
            # './smartbug-dataset/vulnerabilities.json'
            './ge-sc-data/source_code/' + type_key + '/buggy_curated.json'
        ]

        data_vulnerabilities = read_vulnerabilities_json_files(list_vulnerabilities_json_files)

        data_vul_name = list()
        for vul_item in data_vulnerabilities:
            data_vul_name.append(vul_item['name'])

        filePath = './ge-sc-data/source_code/' + type_key + '/clean_' + str(type_vul) + '_buggy_curated_0'
        list_sol_files = os.listdir(filePath)
        count = 0

        # empty_dict = {
        #     "name": "",
        #     "path": "",
        #     "source": "etherscan.io",
        #     "vulnerabilities": []
        # }

        # 遍历文件名，找到不存在bug_json文件的文件
        for sol_item in list_sol_files:
            if sol_item not in data_vul_name:
                # empty_dict["name"] = sol_item
                # empty_dict["path"] = "dataset/"+ type_key + "/" + sol_item
                data_vulnerabilities.append(
                    {
                        "name": sol_item,
                        "path": "dataset/" + type_key + "/" + sol_item,
                        "source": "etherscan.io",
                        "vulnerabilities": []
                    }
                )
                count += 1

        print(type_key, "该类型具有的漏洞文件数:", type_vul)
        print("添加的无漏洞文件数:", count)

        # if count == type_vul:
        write_path = './ge-sc-data/source_code/' + type_key + '/all.json'
        write_vulnerabilities_json_files(data_vulnerabilities, write_path)


def is_empty_line(file_path, line_number):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        if line_number <= len(lines):
            line_content = lines[line_number - 1].strip()
            return line_content == '' or line_content.startswith('//')
        else:
            return False

def filter_vulnerabilities(data, type_key, type_vul):
    err_tot_count = 0
    for item in data:
        filename =  item["name"]
        file_path = f'./ge-sc-data/source_code/' + type_key + '/clean_' + str(type_vul) + f'_buggy_curated_0/{filename}'
        for vulnerability in item["vulnerabilities"]:
            error_list = []
            error_count = 0

            valid_lines = []
            for line in vulnerability["lines"]:
                if is_empty_line(file_path, line):
                    error_count += 1
                    error_list.append(line)
                else:
                    valid_lines.append(line)
            vulnerability["lines"] = valid_lines
            err_tot_count += error_count
            print(f"{filename}=====err_count:{error_count}=======error_labels_list:{error_list}")
    return data, err_tot_count

def del_all_json_label_with_blank():
    for type_key, type_vul in type.items():
        list_vulnerabilities_json_files = [
            './ge-sc-data/source_code/' + type_key + '/all.json'
        ]
        print("===================删除all.json对应的空行label=========处理漏洞类别：",type_key)

        data_vulnerabilities = read_vulnerabilities_json_files(list_vulnerabilities_json_files)

        updated_data , count= filter_vulnerabilities(data_vulnerabilities, type_key, type_vul)

        updata_path = './ge-sc-data/source_code/' + type_key + '/all.json'
        print(f'err_total_count:{count}')
        write_vulnerabilities_json_files(updated_data, updata_path)




def remove_comments_and_empty_lines(source_code):
    # 用空行替换块注释，保留行数
    def replacer(match):
        # 计算块注释中的换行符数量,用相同数量的换行符替换块注释
        return '\n' * match.group(0).count('\n')

    source_code = re.sub(r'/\*.*?\*/', replacer, source_code, flags=re.DOTALL)
    # 正则表达式删除行注释
    source_code = re.sub(r'//.*', '', source_code)
    lines = source_code.split('\n')
    # 将注释转为空行的合约文件
    non_note_lines = lines
    # 删除空行并保留非空行
    non_empty_lines = [line for line in lines if line.strip() != '']
    return non_note_lines, non_empty_lines


def adjust_vulnerability_tags(original_lines, non_empty_lines, vulnerability):
    # 建立原行号到处理后行号的映射
    line_mapping = {}
    non_empty_index = 0

    for original_index, line in enumerate(original_lines):
        if line.strip() != '':  # 跳过空行
            line_mapping[original_index + 1] = non_empty_index + 1
            non_empty_index += 1

    # 调整漏洞标签
    adjusted_vulnerability_lines = list()
    for vul_item in vulnerability['vulnerabilities']:
        new_lines_list = list()
        for line_item in vul_item['lines']:
            new_line = line_mapping[line_item]
            new_lines_list.append(new_line)

        # 得到一个指向新文件的漏洞行index的列表
        adjusted_vulnerability_lines.extend(new_lines_list)

    adjusted_vulnerability_lines.sort()
    return adjusted_vulnerability_lines


def process_contract(file_path, vulnerability_lines):
    with open(file_path, 'r', encoding='utf-8') as file:
        source_code = file.read()

    original_lines = source_code.split('\n')
    non_note_lines, non_empty_lines = remove_comments_and_empty_lines(source_code)
    adjusted_vulnerability_lines = adjust_vulnerability_tags(non_note_lines, non_empty_lines, vulnerability_lines)

    return '\n'.join(non_empty_lines), adjusted_vulnerability_lines


def preprocess_all_file():
    for type_key, type_vul in type.items():
        list_vulnerabilities_json_files = [
            './ge-sc-data/source_code/' + type_key + '/all.json'
        ]
        print("=====================================处理漏洞类别：",type_key)

        data_vulnerabilities = read_vulnerabilities_json_files(list_vulnerabilities_json_files)

        dict_data_vulnerabilities = dict()
        for vul_item in data_vulnerabilities:
            dict_data_vulnerabilities[vul_item['name']] = vul_item

        input_filePath = './ge-sc-data/source_code/' + type_key + '/clean_' + str(type_vul) + '_buggy_curated_0'
        output_filePath = './ge-sc-data/source_code/' + type_key + '/all_clean/'
        list_sol_files = os.listdir(input_filePath)

        for sol_name in list_sol_files:
            sol_filePath = input_filePath + '/' + sol_name
            print(sol_name)
            processed_source_code, adjusted_data_vulnerabilities_lines = process_contract(sol_filePath,
                                                                                          dict_data_vulnerabilities[
                                                                                              sol_name])

            # 将处理后的代码保存到新文件
            with open(output_filePath + sol_name, 'w') as file:
                file.write(processed_source_code)
            print("处理后的智能合约文件已保存。")

            # 修改行标签
            new_data_vulnerabilities = [{
                "lines": adjusted_data_vulnerabilities_lines,
                "category": type_key
            }]
            dict_data_vulnerabilities[sol_name]['vulnerabilities'] = new_data_vulnerabilities
        write_path = './ge-sc-data/source_code/' + type_key + '/preprocess_all.json'
        write_vulnerabilities_json_files(data_vulnerabilities, write_path)


if __name__ == '__main__':
    # 制作所有bug文件的json
    # get_buggy_curated_json()
    # 检查生成的bug文件json
    # check_buggy_curated_json('denial_of_service')

    # 生成无bug和有bug文件的总json, 所有类别的all.json文件已经生成
    # get_all_json()

    # 将all.json中对应空行的标签删去
    # del_all_json_label_with_blank()

    # 预处理合约文件，将注释和空行去除，同时修改漏洞行标签
    preprocess_all_file()
