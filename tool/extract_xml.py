import os
import re
import bz2
from lxml import etree
from tqdm import tqdm

def clean_wiki_text(text):
    """维基文本清洗函数"""
    if not text:
        return ""
    
    # 1. 移除XML注释
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # 2. 分阶段移除嵌套模板（最多处理10层嵌套）
    for _ in range(10):
        new_text = re.sub(r'{{(?:[^{}]|{[^{}]*})*}}', '', text, flags=re.DOTALL)
        if new_text == text:
            break
        text = new_text
    
    # 3. 处理链接（保留显示文本）
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]+)\]\]', r'\1', text)
    
    # 4. 移除HTML标签和引用
    text = re.sub(r'<[^>]+>', '', text)
    
    # 5. 移除表格和画廊
    text = re.sub(r'{\|.*?\|}', '', text, flags=re.DOTALL)
    text = re.sub(r'<gallery>.*?</gallery>', '', text, flags=re.DOTALL)
    
    # 6. 保留中文、基本标点和换行
    text = re.sub(
        r'[^\u4e00-\u9fff，。！？；："“”‘’（）《》【】、\s]', 
        '', 
        text
    )
    
    # 7. 规范化空白字符
    text = re.sub(r'[ \t\r\f\v]+', ' ', text)  # 合并连续空白
    text = re.sub(r'( ?\n ?)+', '\n', text)     # 规范化换行
    
    # 8. 移除短行（小于5字符）和空行
    lines = [line.strip() for line in text.split('\n') 
             if len(line.strip()) >= 5]
    
    return '\n'.join(lines)

def process_single_xml(input_path, output_path):
    """处理单个XML文件"""
    # 支持bz2压缩文件直接读取
    if input_path.endswith('.bz2'):
        open_func = bz2.open
        open_mode = 'rb'
    else:
        open_func = open
        open_mode = 'rb'  
    
    page_count = 0
    with open_func(input_path, open_mode) as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        

        parser = etree.XMLParser(recover=True)
        

        context = etree.iterparse(infile, events=('end',), tag='{*}page')
        

        for event, elem in tqdm(context, desc=f"处理 {os.path.basename(input_path)}"):

            ns = elem.findtext('{*}ns', default='')
            title = elem.findtext('{*}title', default='')
            redirect = elem.find('{*}redirect')
            
            revision = elem.find('.//{*}revision')
            if revision is not None:
                text_elem = revision.find('{*}text')
                text = text_elem.text if text_elem is not None else ''
            else:
                text = ''
  
            if ns == '0' and redirect is None:
                cleaned = clean_wiki_text(text)
                if cleaned:
                    outfile.write(f"【{title}】\n")
                    outfile.write(cleaned + "\n\n")
                    page_count += 1
            
            # 关键：清理内存
            elem.clear()

            while elem.getprevious() is not None:
                del elem.getparent()[0]
        
    return page_count

def process_xml_files(input_dir, output_dir):
    """处理目录中的所有XML文件"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持.xml和.xml.bz2文件
    xml_files = [f for f in os.listdir(input_dir) 
                if f.endswith(('.xml', '.xml.bz2'))]
    
    total_pages = 0
    
    for xml_file in xml_files:
        input_path = os.path.join(input_dir, xml_file)
        output_path = os.path.join(output_dir, 
                                  f"cleaned_{os.path.splitext(os.path.splitext(xml_file)[0])[0]}.txt")

        print(f"处理文件: {xml_file}")
        page_count = process_single_xml(input_path, output_path)
        print(f"已处理页面数: {page_count}")
        total_pages += page_count
    
    print(f"\n处理完成! 总页面数: {total_pages}")

if __name__ == "__main__":
    input_dir = "xml"       # XML文件所在目录
    output_dir = "big_text"     # 清洗后文本输出目录
    
    process_xml_files(input_dir, output_dir)