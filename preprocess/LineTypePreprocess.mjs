import {CompileFailedError, compileSol} from "solc-typed-ast";
import {ASTReader} from "solc-typed-ast";


import fs from 'fs';
import path from 'path'

// 改变漏洞类别
const vul_type = 'unchecked_low_level_calls'

async function CompileContract(filePath) {


    return result
}

async function WriteTypeList(filePath, sourceCode) {

    // 读取Solidity源代码并编译
    try {
        var result = await compileSol(filePath, "auto", []);
        console.log("Compilation successful!");
        console.log(result);
    } catch (error) {
        if (error instanceof CompileFailedError) {
            console.error("Compilation failed:", error.errors);
        } else {
            console.error("An error occurred:", error);
        }
    }


    const reader = new ASTReader();
    const sourceUnits = reader.read(result.data);

// 打印AST节点和对应的行号
    var floor = 0;
    var typedict = {};

    sourceUnits.forEach((sourceUnit) => {
        // console.log(`Source Unit: ${sourceUnit.sourceEntryKey}`);

        const traverseNode = (node) => {
            // console.log(`FLOOR ${floor} ============`)
            if (node.src) {
                const [start, length] = node.src.split(':').map(Number);
                const end = start + length;
                /* 这里sourceCode是把合约中所有字符读取成一个字符串。通过得到的ast节点中的src来切割具体的字符
                    然后使用split('\n')切割成行
                */
                const lines = sourceCode.substring(start, end).split('\n');

                lines.forEach((line, index) => {
                    // (0, start)获取前面有多少行
                    // const lineNumber = sourceCode.substring(0, start).split('\n').length + index + 1;
                    const lineNumber = sourceCode.substring(0, start).split('\n').length + index;
                    // console.log(`Line ${lineNumber}: ${line.trim()}`);
                    // console.log(`Node Type: ${node.type}`);
                    if (!(lineNumber in typedict)) {
                        typedict[lineNumber] = []
                    }
                    typedict[lineNumber].push(node.type)

                });
            }

            if (node.children) {
                floor++;
                node.children.forEach(traverseNode);
            }
            floor--;

        };

        traverseNode(sourceUnit);
        // sourceUnit.children.forEach(traverseNode);
    });
    return typedict
}




// const directory = '../data/ge-sc-data/source_code/access_control/all_clean'
const directory = '../data/ge-sc-data/source_code/'+ vul_type + '/all_clean'

const files = fs.readdirSync(directory);
var nameTypedict = {};
for (const file of files){
    const fullPath = path.join(directory, file);
    const filename = path.basename(fullPath);
    console.log(`File: ${fullPath}`);
    // 读取文件内容
    const sourcecode = fs.readFileSync(fullPath, 'utf-8');
    const typedict = await WriteTypeList(fullPath, sourcecode)

    nameTypedict[filename] = Object.values(typedict);
}
//  files.forEach(async file => {
//     const fullPath = path.join(directory, file);
//     const filename = path.basename(fullPath);
//     console.log(`File: ${fullPath}`);
//     // 读取文件内容
//     const sourcecode = fs.readFileSync(fullPath, 'utf-8');
//     const typedict = await WriteTypeList(fullPath, sourcecode)
//
//     nameTypedict[filename] = typedict
// });

// 将对象转换为 JSON 字符串
const jsonString = JSON.stringify(nameTypedict, null, 2);

// 写入文件， 改动漏洞类别
// const filePath = '../data/ge-sc-data/source_code/access_control/typeList.json'
const filePath = '../data/ge-sc-data/source_code/'+ vul_type + '/typeList.json'

fs.writeFile(filePath, jsonString, 'utf8', (err) => {
    if (err) {
        console.error('An error occurred while writing to file:', err);
    } else {
        console.log('File has been saved successfully.');
    }
});


// console.log("Used compiler version: " + result.compilerVersion);
// console.log(sourceUnits[0].print());