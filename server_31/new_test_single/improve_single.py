import re
import os
import numpy as np


class CreateGetTrainBatchFile(object):
    def __init__(self, single_file_name='single.py', new_file_name='data_samping.py'):
        self.single_file_name = single_file_name
        self.new_file_name = new_file_name
        self.func_name = None
        self.func_param = None
        self.feed_list = []
        self.statement_list = []
        self.run_code_count = 0
        self.func_count = 0
        self.spaces_num_list = []  # -1:已注释，-2：空行，-3：类，-4：内嵌函数，-5：新注释，其余为空格数
        self.spaces_content_list = []
        self.all_variate_list = []
        self.all_useful_param_list = []
        self.next_set_flush_count = 0
        self.entrance_count = 0
        self.del_block_head_count_list = []

    def write_off_statement(self, count):
        line_list = self.statement_list[count].split('\n')[:-1]
        for i in range(len(line_list)):
            line_list[i] = '#' + line_list[i]
        statement = '\n'.join(line_list)+'\n'
        self.statement_list[count] = statement

    def uncomment_statement(self, count):
        line_list = self.statement_list[count].split('\n')[:-1]
        for i in range(len(line_list)):
            line_list[i] = '#'.join(line_list[i].split('#')[1:])
        statement = '\n'.join(line_list) + '\n'
        self.statement_list[count] = statement

    # 1、将single文件分成语句存入列表
    def single_divide_into_statement(self):
        left_bracket_list = ['(', '[', '{']
        right_bracket_list = [')', ']', '}']
        bracket_num = 0
        single_quotes_num = 0
        double_quotes_num = 0
        statement = ''
        with open(self.single_file_name, 'r', encoding="utf-8") as file_handle:
            for line in file_handle:
                statement += line
                real_line = re.sub('[\'\"]+?[\s\S]*?[\'\"]+?', ' ', line).split('#')[0]
                # 匹配三引号
                single_quotes_num += real_line.count('\'\'\'')
                double_quotes_num += real_line.count('\"\"\"')
                if single_quotes_num + double_quotes_num > 0:  # 三引号存在
                    if single_quotes_num % 2 == 1 or double_quotes_num % 2 == 1:
                        continue
                    else:
                        self.statement_list.append(statement)
                        statement = ''
                        single_quotes_num = double_quotes_num = 0
                        continue
                # 匹配括号
                for left_bracket in left_bracket_list:
                    bracket_num += real_line.count(left_bracket)
                for right_bracket in right_bracket_list:
                    bracket_num -= real_line.count(right_bracket)
                # 匹配反斜杠
                backslash_num = real_line.count('\\\n')
                if backslash_num == 1:
                    continue
                if bracket_num == 0:
                    self.statement_list.append(statement)
                    statement = ''

    # 2、找出run语句，记录位置，返回数据
    def find_feed_list(self, feed_dicts, count_list):
        feed_list = []
        is_still_find = True
        while is_still_find:
            feed_dicts = feed_dicts.strip()
            if feed_dicts.startswith('{'):
                feed_dicts = re.findall(re.compile(r'\{([\s\S]*?)\}'), feed_dicts)
                feed_list = feed_dicts[0].split(",")
                for i in range(len(feed_list)):
                    feed_list[i] = feed_list[i][feed_list[i].find(':') + 1:].strip()
                is_still_find = False
                self.all_useful_param_list += feed_list
            else:
                pattern_new_feed_dicts = re.compile(r'%s\s*=([\s\S]*)' % feed_dicts)
                is_find = False
                count_list_length = len(count_list)
                for _ in range(count_list_length):
                    count = count_list.pop(0)
                    statement = self.statement_list[count]
                    new_feed_dicts_list = re.findall(pattern_new_feed_dicts, statement.split('#')[0].strip())
                    if len(new_feed_dicts_list) > 0:
                        feed_dicts = new_feed_dicts_list[0]
                        self.all_useful_param_list.append(feed_dicts)
                        self.write_off_statement(count)
                        self.spaces_num_list[count] = -1
                        is_find = True
                        break
                    else:
                        count_list.append(count)
                if not is_find:
                    is_still_find = False
        return feed_list

    def get_run_message(self):
        pattern_run_statement = re.compile(r'([\S\s]*?)\.\s*run\s*\(([\S\s]*?)feed_dict\s*=([\S\s]*)\)')
        pattern_space = re.compile(r'(\s*?)\S')
        run_statement_list = []
        entrance_list = []
        # 找到每段语句的起始空格数
        for i in range(len(self.statement_list)):
            if self.statement_list[i].strip().startswith('#'):
                self.spaces_num_list.append(-1)
                self.spaces_content_list.append('')
                continue  # 忽略已注释的行
            else:
                spaces_num = re.findall(pattern_space, self.statement_list[i])
                if spaces_num:
                    spaces_content = spaces_num[0]
                    tab_num = spaces_num[0].count('\t')
                    spaces_num = len(spaces_num[0]) + 3 * tab_num
                else:
                    spaces_num = -2  # 空行
                    spaces_content = ''
                self.spaces_num_list.append(spaces_num)
                self.spaces_content_list.append(spaces_content)
        # 忽略类
        class_spaces = None
        pattern_class = re.compile(r'^\s*class\s*[\\\n]?\s*\S')
        for count in range(len(self.statement_list)):
            if self.spaces_num_list[count] < 0:
                continue  # 忽视注释行与空行
            if not class_spaces:
                class_list = re.findall(pattern_class, self.statement_list[count])
                if class_list:
                    class_spaces = self.spaces_num_list[count]
                    self.spaces_num_list[count] = -3
            else:
                if self.spaces_num_list[count] > class_spaces:
                    self.spaces_num_list[count] = -3
                else:
                    class_spaces = None
                    class_list = re.findall(pattern_class, self.statement_list[count])
                    if class_list:
                        class_spaces = self.spaces_num_list[count]
                        self.spaces_num_list[count] = -3
        # 忽略内嵌函数
        inline_func_space = None
        pattern_inline_func = re.compile(r'^\s+def\s*[\\\n]?\s*\S')
        for count in range(len(self.statement_list)):
            if self.spaces_num_list[count] < 0:
                continue  # 忽略注释行、空行与类
            if not inline_func_space:
                inline_func_list = re.findall(pattern_inline_func, self.statement_list[count])
                if inline_func_list:
                    inline_func_space = self.spaces_num_list[count]
                    self.spaces_num_list[count] = -4
            else:
                if self.spaces_num_list[count] > inline_func_space:
                    self.spaces_num_list[count] = -4
                else:
                    inline_func_space = None
                    inline_func_list = re.findall(pattern_inline_func, self.statement_list[count])
                    if inline_func_list:
                        inline_func_space = self.spaces_num_list[count]
                        self.spaces_num_list[count] = -4
        # 找到run_code
        for count in range(len(self.statement_list)):
            if self.spaces_num_list[count] < 0:
                continue  # 忽略注释行、空行、类、内嵌函数
            run_statement_list = re.findall(pattern_run_statement, self.statement_list[count])
            if len(run_statement_list) > 0:
                self.run_code_count = count
                break
        if len(run_statement_list) == 0:  # 文件内无run_code
            print('The document training model does not meet the requirements')
            exit(1)
        # 找到run_code所在函数列号
        for count in range(self.run_code_count, 0, -1):
            if self.spaces_num_list[count] == 0:
                self.func_count = count
                break
        # 找到函数后第一个顶格列号
        for count in range(self.run_code_count, len(self.statement_list)):
            if self.spaces_num_list[count] == 0:
                self.next_set_flush_count = count
                break
        # 找到文件入口列号
        pattern_entrance = re.compile(r'^if\s+__name__\s*==\s*[\'\"]+?__main__[\'\"]+?\s*:')
        for count in range(len(self.statement_list)-1, 0, -1):
            entrance_list = re.findall(pattern_entrance, self.statement_list[count])
            if entrance_list:
                self.entrance_count = count
                break
        if len(entrance_list) == 0:  # 文件内无入口
            print('The document no entry')
            exit(1)
        # 找到函数名和函数的参数
        pattern_def = re.compile(r'def\s+(\S*?)\s*\(([\s\S]*?)\)')
        func_message_list = re.findall(pattern_def, self.statement_list[self.func_count])
        if len(func_message_list) > 0:
            self.func_name = func_message_list[0][0]
            self.func_param = func_message_list[0][1]
        else:
            print('The training model code is not in the function')
            exit(1)
        # 找到训练数据
        if run_statement_list[0][2] == "":
            print('The feeding data mode does not meet the requirements')
            exit(1)
        feed_dicts = run_statement_list[0][2]
        find_variate_count_list = []
        for count in range(self.run_code_count-1, self.func_count, -1):
            if self.spaces_num_list[count] >= 0:
                find_variate_count_list.append(count)  # 倒序
        feed_list = self.find_feed_list(feed_dicts, find_variate_count_list)
        if feed_list:
            self.feed_list = feed_list
        else:
            print('The data is not in the function')
            exit(1)
        # 返回数据，注释训练代码
        return_content = '\n'+self.spaces_content_list[self.run_code_count]+'return %s\n' % self.feed_list
        return_content = return_content.replace("'", "")
        run_code_line_list = self.statement_list[self.run_code_count].split('\n')[:-1]
        for _ in range(len(run_code_line_list)):
            new_line = '#'+run_code_line_list[0]
            run_code_line_list.append(new_line)
            run_code_line_list.pop(0)
        run_code = '\n'.join(run_code_line_list)
        run_code += return_content
        self.statement_list[self.run_code_count] = run_code

    # 3、更改循环，找到step列表
    def change_cycle_lines(self):
        pattern_for = re.compile(r'for \s*([\s\S]*?) \s*in \s*([\S\s]*?):')
        pattern_while = re.compile(r'while \s*([\s\S]*?) \s*< \s*([\S\s]*?):')
        cycle_count_list = []
        step_name_list = []
        # 找到函数内所有循环的列号
        for i in range(self.func_count+1, self.run_code_count):
            if self.spaces_num_list[i] < 0:
                continue  # 忽略注释行、空行、类、内嵌函数
            else:
                message_for = re.findall(pattern_for, self.statement_list[i])
                message_while = re.findall(pattern_while, self.statement_list[i])
                if len(message_for) > 0:
                    cycle_count_list.append(i)
                    step_name_list.append(message_for[0][0])
                elif len(message_while) > 0:
                    cycle_count_list.append(i)
                    step_name_list.append(message_while[0][0])
        # 如果run语句在循环内则记住循环列号
        if cycle_count_list:
            for _ in range(len(cycle_count_list)):
                is_run_in_cycle = True
                for i in range(cycle_count_list[0] + 1, self.run_code_count + 1):
                    if self.spaces_num_list[i] < 0:
                        continue  # 忽略注释行、空行、类、内嵌函数
                    elif self.spaces_num_list[i] <= self.spaces_num_list[cycle_count_list[0]]:
                        is_run_in_cycle = False
                        break
                if is_run_in_cycle:
                    count = cycle_count_list[0]
                    step_name = step_name_list[0]
                    cycle_count_list.append(count)
                    step_name_list.append(step_name)
                cycle_count_list.pop(0)
                step_name_list.pop(0)
        # 更改循环
        if cycle_count_list:
            for cycle_count in cycle_count_list:
                self.statement_list[cycle_count] = self.spaces_content_list[cycle_count]+'if True:\n'
        # 注释循环中的break和continue语句
        for cycle_count in cycle_count_list:
            for count in range(cycle_count+1, self.next_set_flush_count):
                if self.spaces_num_list[count] < 0:  # 忽略注释行、空行、类、内嵌函数
                    continue
                if self.spaces_num_list[count] <= self.spaces_num_list[cycle_count]:  # 超出循环
                    break
                pattern_break = re.compile(r'^\s+break')
                pattern_continue = re.compile(r'^\s+continue')
                break_list = re.findall(pattern_break, self.statement_list[count])
                continue_list = re.findall(pattern_continue, self.statement_list[count])
                if len(break_list)+len(continue_list) > 0:
                    self.statement_list[count] = self.spaces_content_list[count]+'return False\n'
        return cycle_count_list, step_name_list

    # 4、将函数内语句归为若干语句块加入列表
    def mark_off_block(self):
        block_count_list = []
        func_count_list = []
        spaces_level_list = []
        for i in range(self.func_count+1, self.next_set_flush_count):
            spaces = self.spaces_num_list[i]
            if spaces >= 0:  # 忽略空行和注释行
                block_count_list.insert(0, i)  # 倒序
                func_count_list.append(i)
                spaces_level_list.append(spaces)
        spaces_level_list = sorted(list(set(spaces_level_list)), reverse=True)
        part_block_count_list = []
        block_head_count_list = []
        for i in range(len(spaces_level_list)-1):
            for _ in range(len(block_count_list)):
                count = block_count_list[0]
                if isinstance(count, list):
                    part_block_count_list.insert(0, count)  # 块的内容
                else:
                    if self.spaces_num_list[count] == spaces_level_list[i]:  # 块的内容
                        part_block_count_list.insert(0, count)
                    elif self.spaces_num_list[count] == spaces_level_list[i+1]:  # 块的开头
                        if len(part_block_count_list) > 0:
                            block_head_count_list.insert(0, count)
                            part_block_count_list.insert(0, count)
                            block_count_list.append(part_block_count_list)
                            part_block_count_list = []
                        else:
                            block_count_list.append(count)
                    else:
                        block_count_list.append(count)
                block_count_list.pop(0)
        block_count_list.reverse()
        return block_count_list, block_head_count_list

    # 5、注释函数内多余行
    def split_variate(self, variate_list, symbol):
        variate_list_length = len(variate_list)
        for _ in range(variate_list_length):
            if variate_list[0] == '\n':
                variate_list.pop(0)
                continue
            new_variate_list = variate_list[0].split(symbol)
            for new_variate in new_variate_list:
                if new_variate != '':
                    variate_list.append(new_variate)
            variate_list.pop(0)
        return list(set(variate_list))

    def find_new_variate(self, variate_list, count_list, cancel=False):
        useful_count_list = []
        variate_list_length = len(variate_list)
        for _ in range(variate_list_length):
            variate = variate_list.pop(0)
            self.all_variate_list.append(variate)
            count_list_length = len(count_list)
            for _ in range(count_list_length):
                count = count_list.pop(0)
                if cancel:
                    statement = self.statement_list[count][1:].split('#')[0].strip()
                else:
                    statement = self.statement_list[count].split('#')[0].strip()
                if '=' not in statement:
                    count_list.append(count)
                    continue
                statement_left = statement.split('=')[0]
                statement_right = '='.join(statement.split('=')[1:])
                pattern_new_variate_1 = re.compile(r'^%s\s*$' % variate)  # 单个变量名
                pattern_new_variate_2 = re.compile(r'^%s\s*,[\S\s]*?' % variate)  # 多个变量名顶格
                pattern_new_variate_3 = re.compile(r'[\S\s]*?,\s*%s\s*,[\S\s]*?' % variate)  # 多个变量名中间
                pattern_new_variate_4 = re.compile(r'[\S\s]*?,\s*%s\s*$' % variate)  # 多个变量名末尾
                new_variate_list_1 = re.findall(pattern_new_variate_1, statement_left)
                new_variate_list_2 = re.findall(pattern_new_variate_2, statement_left)
                new_variate_list_3 = re.findall(pattern_new_variate_3, statement_left)
                new_variate_list_4 = re.findall(pattern_new_variate_4, statement_left)
                if len(new_variate_list_1)+len(new_variate_list_2)+len(new_variate_list_3)+len(new_variate_list_4) > 0:
                    new_variate = statement_right
                    variate_list.append(new_variate)
                    useful_count_list.append(count)
                else:
                    count_list.append(count)
        return useful_count_list, count_list

    def del_block_count(self, block_count_list, count):
        for block in block_count_list:
            if isinstance(block, list):
                self.del_block_count(block, count)
            else:
                if block == count:
                    block_count_list.remove(block)
                    break
                elif block > count:
                    break

    def del_block_head_count(self, block_count_list):
        block_count_list_length = len(block_count_list)
        for _ in range(block_count_list_length):
            block = block_count_list.pop(0)
            if isinstance(block, list):
                if len(block) == 1:
                    self.del_block_head_count_list.append(block[0])
                    continue
                self.del_block_head_count(block)
            block_count_list.append(block)

    def write_off_func_lines(self, block_count_list, block_head_count_list, cycle_count_list):
        symbol_list = ['(', ')', ',', '[', ']', ':', '+', '-', '*', '/', '%', '//', '.', '=', ' ']
        variate_list = []
        find_variate_count_list = []
        stay_count_list = []
        for variate in self.feed_list:
            variate_list.append(variate)
        for count in range(self.run_code_count - 1, self.func_count, -1):
            if self.spaces_num_list[count] >= 0:
                find_variate_count_list.append(count)  # 倒序
        # 将"with tf.Session() as sess:"替换为"if True:"
        pattern_graph = re.compile(r'with\s+\S+\s*\.Session\s*\([\S\s]*\)')
        block_head_sess_count_list = []
        for block_count in block_head_count_list:
            if block_count > self.run_code_count:
                continue
            statement = self.statement_list[block_count]
            graph_list = re.findall(pattern_graph, statement.split('#')[0].strip())
            if len(graph_list) > 0:
                block_head_sess_count_list.append(block_count)
                self.statement_list[block_count] = self.spaces_content_list[block_count]+"if True:\n"

        while len(variate_list) > 0:
            for symbol in symbol_list:
                variate_list = self.split_variate(variate_list, symbol)
            self.all_useful_param_list += variate_list
            self.all_useful_param_list = list(set(self.all_useful_param_list))
            useful_count_list, find_variate_count_list = self.find_new_variate(variate_list, find_variate_count_list)
            if useful_count_list:
                for count in useful_count_list:
                    stay_count_list.append(count)
        stay_count_list.sort(reverse=True)
        del_count_list = []
        for count in find_variate_count_list:
            if count not in block_head_count_list:
                del_count_list.append(count)
        del_count_list.sort()
        for count in del_count_list:
            self.write_off_statement(count)  # 注释多余的行
            self.spaces_num_list[count] = -5  # 将注释行空格数改为-5
            self.del_block_count(block_count_list, count)
        # 注释只剩block_head的块
        for _ in range(5):
            self.del_block_head_count(block_count_list)
        for count in self.del_block_head_count_list:
            self.write_off_statement(count)
        # 将未被注释的block_head中的参数赋值行取消注释
        useful_block_head_count_list = []
        for count in block_head_count_list:
            if count in (self.del_block_head_count_list+cycle_count_list+block_head_sess_count_list):
                continue
            useful_block_head_count_list.append(count)
        useful_block_head_count_list_length = len(useful_block_head_count_list)
        for _ in range(useful_block_head_count_list_length):
            symbol_list = ['(', ')', ',', '[', ']', ':', '+', '-', '*', '/', '%', '//', '.', '=', ' ', '<', '>']
            find_variate_count_list = []
            stay_count_list = []
            block_head_count = useful_block_head_count_list.pop(0)
            for count in range(self.func_count+1, block_head_count):
                if self.spaces_num_list[count] == -5:
                    find_variate_count_list.append(count)
            variate_list = [self.statement_list[block_head_count].strip()]
            while len(variate_list) > 0:
                for symbol in symbol_list:
                    variate_list = self.split_variate(variate_list, symbol)
                self.all_useful_param_list += variate_list
                self.all_useful_param_list = list(set(self.all_useful_param_list))
                useful_count_list, find_variate_count_list = self.find_new_variate(variate_list,
                                                                                   find_variate_count_list, True)
                if useful_count_list:
                    for count in useful_count_list:
                        stay_count_list.append(count)
            stay_count_list.sort(reverse=True)
            if stay_count_list:
                for stay_count in stay_count_list:
                    self.uncomment_statement(stay_count)
                    spaces_content = self.spaces_content_list[stay_count]
                    tab_num = spaces_content.count('\t')
                    spaces_num = len(spaces_content) + 3 * tab_num
                    self.spaces_num_list[stay_count] = spaces_num

    # 6、确保循环取值
    def ensure_cycle(self, cycle_count_list, step_name_list):
        useful_step_name_list = []
        symbol_list = ['(', ')', ',', '[', ']', ':', '+', '-', '*', '/', '%', '//', '.', '=', ' ', '{', '}']
        for symbol in symbol_list:
            self.all_useful_param_list = self.split_variate(self.all_useful_param_list, symbol)
        if step_name_list:
            for step in step_name_list:
                if step in self.all_useful_param_list:
                    useful_step_name_list.append(step)
            if useful_step_name_list:
                step_name = useful_step_name_list[-1]
            else:
                # 情况一：step不在循环语句中,在循环语句块中寻找赋值语句
                step_assignment_list = []
                all_cycle_count_list = []
                for cycle_count in cycle_count_list:
                    for count in range(cycle_count, self.next_set_flush_count):
                        if self.spaces_num_list[count] > self.spaces_num_list[cycle_count]:
                            all_cycle_count_list.append(count)
                all_cycle_count_list = sorted(list(set(all_cycle_count_list)))
                pattern_step_assignment_1 = re.compile(r'^\s*(\S+?)\s*\+=')
                pattern_step_assignment_2 = re.compile(r'^\s*(\S+?)\s*\-=')
                pattern_step_assignment_3 = re.compile(r'^\s*(\S+?)\s*\*=')
                pattern_step_assignment_4 = re.compile(r'^\s*(\S+?)\s*/=')
                for count in all_cycle_count_list:
                    step_assignment_1 = re.findall(pattern_step_assignment_1, self.statement_list[count])
                    step_assignment_2 = re.findall(pattern_step_assignment_2, self.statement_list[count])
                    step_assignment_3 = re.findall(pattern_step_assignment_3, self.statement_list[count])
                    step_assignment_4 = re.findall(pattern_step_assignment_4, self.statement_list[count])
                    if step_assignment_1:
                        step_assignment_list.append(step_assignment_1[0])
                    elif step_assignment_2:
                        step_assignment_list.append(step_assignment_2[0])
                    elif step_assignment_3:
                        step_assignment_list.append(step_assignment_3[0])
                    elif step_assignment_4:
                        step_assignment_list.append(step_assignment_4[0])
                if step_assignment_list:
                    step_name = step_assignment_list[-1]
                # 情况二：采用global形式传入函数
                else:
                    is_global_introduction = False
                    pattern_global_step = re.compile(r'^#\s*global\s+[\S\s]+')
                    for count in range(self.func_count+1, self.next_set_flush_count):
                        if self.spaces_num_list[count] == -1:
                            global_param_list = re.findall(pattern_global_step, self.statement_list[count])
                            if global_param_list:
                                global_step_list = global_param_list[0].split(',')
                                global_step_list_length = len(global_step_list)
                                for _ in range(global_step_list_length):
                                    global_step = global_step_list.pop(0)
                                    if global_step.strip() != '':
                                        global_step_list.append(global_step.strip())
                                for global_step in global_step_list:
                                    if global_step in self.all_useful_param_list:
                                        is_global_introduction = True
                                        self.uncomment_statement(count)
                    if not is_global_introduction:
                        step_name = "step_tesra"
        else:
            '''还有其他情况没有step_name
            '''
            step_name = "step_tesra"
        statement = self.statement_list[self.func_count].split('#')[0]
        if self.func_param:
            new_statement = ')'.join(statement.split(')')[:-1]) + ', %s=0):\n' % step_name
        else:
            new_statement = ')'.join(statement.split(')')[:-1]) + '%s=0):\n' % step_name
        self.statement_list[self.func_count] = new_statement
        # 注释给step_name赋值语句
        if step_name:
            pattern_step_assignment = re.compile(r'^\s*%s\s*[\+\-\*/]?=' % step_name)
            for count in range(self.func_count+1, self.run_code_count):
                if self.spaces_num_list[count] >= 0:
                    step_assignment_list = re.findall(pattern_step_assignment, self.statement_list[count])
                    if step_assignment_list:
                        self.write_off_statement(count)

    # 7、生成启动函数get_train_batch_tesra()
    def create_produce_data_func(self):
        symbol_list = ['(', ')', ',', '[', ']', ':', '+', '-', '*', '/', '%', '//', '.', '=', ' ']
        useful_start_param_list = []
        useless_param_sub_list = []
        useless_start_param_list = []
        start_param = ''
        call_func_count = None
        if self.func_param:
            pattern_start_param = re.compile(r'%s\s*\(([\S\s]*)\)' % self.func_name)
            for count in range(self.entrance_count, len(self.statement_list)):
                start_param_list = re.findall(pattern_start_param, self.statement_list[count])
                if start_param_list:
                    start_param = start_param_list[0]
                    call_func_count = count
                    break
            if not call_func_count:
                print('The function is not called')
                exit(1)
            func_param_list = self.func_param.split(',')
            for i in range(len(func_param_list)):
                if func_param_list[i].strip() not in self.all_useful_param_list:
                    useless_param_sub_list.append(i)  # 不需要的函数下表列表
            start_param_list = start_param.split(',')
            for sub in useless_param_sub_list:
                useless_start_param_list.append(start_param_list[sub].strip())
            for param in start_param_list:
                if param.strip() not in useless_start_param_list:
                    useful_start_param_list.append(param.strip())
            start_code = 4 * ' ' + 'data_list_tesra = %s(%s, step)\n' % (self.func_name, start_param)
            if useless_start_param_list:
                for param in useless_start_param_list:
                    start_code = '    %s = 0\n' % param + start_code
            variate_list = useful_start_param_list  # 有用的启动参数列表
            if variate_list:
                find_variate_count_list = []
                stay_count_list = []
                for count in range(self.entrance_count+1, call_func_count):
                    if self.spaces_num_list[count] >= 0:
                        find_variate_count_list.append(count)
                while len(variate_list) > 0:
                    for symbol in symbol_list:
                        variate_list = self.split_variate(variate_list, symbol)
                    useful_count_list, find_variate_count_list = self.find_new_variate(variate_list,
                                                                                       find_variate_count_list)
                    if useful_count_list:
                        for count in useful_count_list:
                            stay_count_list.append(count)
                if stay_count_list:
                    stay_count_list.sort()
                    add_start_code = ''
                    for count in stay_count_list:
                        add_start_code += 4*' '+self.statement_list[count].lstrip()
                    start_code = add_start_code+start_code
        else:
            start_code = 4 * ' ' + 'data_list_tesra = %s(step)\n' % self.func_name
        start_code = 'import numpy as np\n\n\ndef get_train_batch_tesra(step=0):\n' + start_code
        self.statement_list.insert(self.entrance_count, start_code)
        self.statement_list.insert(self.entrance_count+1, '    return data_list_tesra\n\n\n')
        self.write_to_new_file()

    # 8、通过维度，识别返回值并调整返回值顺序
    def rewrite_file_by_shape(self):
        import data_samping
        data_list = data_samping.get_train_batch_tesra()
        keep_prob_value = 0.5
        keep_prob_sub = 2
        if len(data_list) == 3:
            for i in range(3):
                if isinstance(data_list[i], float) or isinstance(data_list[i], int):
                    keep_prob_value = data_list[i]
                    keep_prob_sub = i
            data_list.pop(keep_prob_sub)
            self.feed_list.pop(keep_prob_sub)
        shape_0 = np.array(data_list[0]).shape
        shape_1 = np.array(data_list[1]).shape
        shape_0_value = 1
        for i in shape_0:
            shape_0_value *= i
        shape_1_value = 1
        for i in shape_1:
            shape_1_value *= i
        if shape_0_value >= shape_1_value:
            pass
        else:
            self.feed_list[0], self.feed_list[1] = self.feed_list[1], self.feed_list[0]
        statement = 4 * ' ' + 'batch_x = np.array(list(data_list_tesra[0]))\n'
        statement += 4 * ' ' + 'batch_y = np.array(list(data_list_tesra[1]))\n'
        statement += 4 * ' ' + 'return batch_x, batch_y\n\n\n'
        statement += 'keep_prob_value = %f\n\n\n' % keep_prob_value
        self.statement_list[self.entrance_count+1] = statement
        self.write_to_new_file()

    def write_to_new_file(self):
        # 将改好的代码写入文件
        file_data = ''
        for statement in self.statement_list:
            file_data += statement
        with open(self.new_file_name, 'w', encoding="utf-8") as f:
            f.write(file_data)

    def run(self):
        self.single_divide_into_statement()
        self.get_run_message()
        cycle_count_list, step_name_list = self.change_cycle_lines()
        block_count_list, block_head_count_list = self.mark_off_block()
        self.write_off_func_lines(block_count_list, block_head_count_list, cycle_count_list)
        self.ensure_cycle(cycle_count_list, step_name_list)
        self.create_produce_data_func()
        self.rewrite_file_by_shape()
        return True


class CreateSaveModelFile(object):
    def __init__(self, single_file_name='single.py', new_file_name='save_model.py',
                 checkpoint_path=r'./model/model.ckpt'):
        self.single_file_name = single_file_name
        self.new_file_name = new_file_name
        self.checkpoint_path = checkpoint_path
        self.sess_name = 'sess'
        self.func_name = None
        self.statement_list = []
        self.run_code = ''
        self.run_code_count = 0
        self.func_count = 0
        self.step_name_list = []
        self.spaces_num_list = []  # -1:已注释，-2：空行，-3：类，-4：内嵌函数，-5：新注释，其余为空格数
        self.spaces_content_list = []
        self.next_set_flush_count = 0
        self.entrance_count = 0

    # 1、将single文件分成语句存入列表
    def single_divide_into_statement(self):
        left_bracket_list = ['(', '[', '{']
        right_bracket_list = [')', ']', '}']
        bracket_num = 0
        single_quotes_num = 0
        double_quotes_num = 0
        statement = ''
        with open(self.single_file_name, 'r', encoding="utf-8") as file_handle:
            for line in file_handle:
                statement += line
                real_line = re.sub('[\'\"]+?[\s\S]*?[\'\"]+?', ' ', line).split('#')[0]
                # 匹配三引号
                single_quotes_num += real_line.count('\'\'\'')
                double_quotes_num += real_line.count('\"\"\"')
                if single_quotes_num+double_quotes_num > 0:  # 三引号存在
                    if single_quotes_num % 2 == 1 or double_quotes_num % 2 == 1:
                        continue
                    else:
                        self.statement_list.append(statement)
                        statement = ''
                        single_quotes_num = double_quotes_num = 0
                        continue
                # 匹配括号
                for left_bracket in left_bracket_list:
                    bracket_num += real_line.count(left_bracket)
                for right_bracket in right_bracket_list:
                    bracket_num -= real_line.count(right_bracket)
                # 匹配反斜杠
                backslash_num = real_line.count('\\\n')
                if backslash_num == 1:
                    continue
                if bracket_num == 0:
                    self.statement_list.append(statement)
                    statement = ''

    # 2、找出run语句，记录位置，并识别sess的名称
    def get_run_message(self):
        pattern_run_statement = re.compile(r'([\S\s]*)\.\s*run\s*\(([\S\s]*?)feed_dict\s*=([\S\s]*)\)')
        pattern_space = re.compile(r'(\s*?)\S')
        run_statement_list = []
        # 找到每段语句的起始空格数
        for i in range(len(self.statement_list)):
            if self.statement_list[i].strip().startswith('#'):
                self.spaces_num_list.append(-1)
                self.spaces_content_list.append('')
                continue  # 忽略已注释的行
            else:
                spaces_num = re.findall(pattern_space, self.statement_list[i])
                if spaces_num:
                    spaces_content = spaces_num[0]
                    tab_num = spaces_num[0].count('\t')
                    spaces_num = len(spaces_num[0])+3 * tab_num
                else:
                    spaces_num = -2  # 空行
                    spaces_content = ''
                self.spaces_num_list.append(spaces_num)
                self.spaces_content_list.append(spaces_content)
        # 忽略类
        class_spaces = None
        pattern_class = re.compile(r'^\s*class\s*[\\\n]?\s*\S')
        for count in range(len(self.statement_list)):
            if self.spaces_num_list[count] < 0:
                continue  # 忽视注释行与空行
            if not class_spaces:
                class_list = re.findall(pattern_class, self.statement_list[count])
                if class_list:
                    class_spaces = self.spaces_num_list[count]
                    self.spaces_num_list[count] = -3
            else:
                if self.spaces_num_list[count] > class_spaces:
                    self.spaces_num_list[count] = -3
                else:
                    class_spaces = None
                    class_list = re.findall(pattern_class, self.statement_list[count])
                    if class_list:
                        class_spaces = self.spaces_num_list[count]
                        self.spaces_num_list[count] = -3
        # 忽略内嵌函数
        inline_func_space = None
        pattern_inline_func = re.compile(r'^\s+def\s*[\\\n]?\s*\S')
        for count in range(len(self.statement_list)):
            if self.spaces_num_list[count] < 0:
                continue  # 忽略注释行、空行与类
            if not inline_func_space:
                inline_func_list = re.findall(pattern_inline_func, self.statement_list[count])
                if inline_func_list:
                    inline_func_space = self.spaces_num_list[count]
                    self.spaces_num_list[count] = -4
            else:
                if self.spaces_num_list[count] > inline_func_space:
                    self.spaces_num_list[count] = -4
                else:
                    inline_func_space = None
                    inline_func_list = re.findall(pattern_inline_func, self.statement_list[count])
                    if inline_func_list:
                        inline_func_space = self.spaces_num_list[count]
                        self.spaces_num_list[count] = -4
        # 找到run_code
        for count in range(len(self.statement_list)):
            if self.spaces_num_list[count] < 0:
                continue  # 忽略注释行、空行、类、内嵌函数
            run_statement_list = re.findall(pattern_run_statement, self.statement_list[count])
            if len(run_statement_list) > 0:
                self.run_code = self.statement_list[count]
                self.run_code_count = count
                break
        if len(run_statement_list) == 0:  # 文件内无run_code
            print('The document training model does not meet the requirements')
            exit(1)
        # 找到run_code所在函数列号
        for count in range(self.run_code_count, 0, -1):
            if self.spaces_num_list[count] == 0:
                self.func_count = count
                break
        # 找到函数后第一个顶格列号
        for count in range(self.run_code_count, len(self.statement_list)):
            if self.spaces_num_list[count] == 0:
                self.next_set_flush_count = count
                break
        # 找到文件入口列号
        pattern_entrance = re.compile(r'^if \s*__name__\s*==\s*[\'\"]+?__main__[\'\"]+?\s*:')
        for count in range(len(self.statement_list) - 1, 0, -1):
            entrance_list = re.findall(pattern_entrance, self.statement_list[count])
            if entrance_list:
                self.entrance_count = count
                break
        # 找到函数名
        pattern_def = re.compile(r'def \s*(\S*?)\s*\([\s\S]*?\)')
        func_name_list = re.findall(pattern_def, self.statement_list[self.func_count])
        if len(func_name_list) > 0:
            self.func_name = func_name_list[0]
        else:
            print('The training model code is not in the function')
            exit(1)
        # 找到session变量名
        if run_statement_list[0][1] == '':  # run语句中无sess
            for statement in self.statement_list:
                if statement.strip().startswith('#'):
                    continue  # 忽略已注释的行
                pattern_sess_name = re.compile(r'(\S+)\s*=\s*\S+?\s*\.\s*InteractiveSession\s*\(')
                sess_name_list = re.findall(pattern_sess_name, statement)
                if len(sess_name_list) > 0:
                    self.sess_name = sess_name_list[0]
        else:
            pattern_session_name = re.compile(r'session\s*=\s*(\S+?)\s*,')
            session_name = re.findall(pattern_session_name, run_statement_list[0][1])
            if len(session_name) > 0:  # sess在括号内
                self.sess_name = session_name[0]
            else:  # sess在括号外
                self.sess_name = run_statement_list[0][0].split('=')[-1].strip()

    # 3、在run_code语句后添加保存操作，更改循环
    def write_off_lines(self):
        pattern_for = re.compile(r'for \s*(\S*?)\s+in \s*([\S\s]*?):')
        pattern_while = re.compile(r'while \s*(\S*?)\s+< \s*([\S\s]*?):')
        for_cycle_count_list = []
        while_cycle_count_list = []
        # 找到函数内run_code之前所有循环的列号
        for count in range(self.func_count+1, self.run_code_count):
            if self.spaces_num_list[count] < 0:
                continue  # 忽略注释行、空行、类、内嵌函数
            else:
                message_for = re.findall(pattern_for, self.statement_list[count])
                message_while = re.findall(pattern_while, self.statement_list[count])
                if len(message_for) > 0:
                    for_cycle_count_list.append(count)
                elif len(message_while) > 0:
                    while_cycle_count_list.append(count)
                    self.step_name_list.append(message_while[0][0])
        # 添加保存操作
        run_statement = self.statement_list[self.run_code_count]
        spaces = self.spaces_content_list[self.run_code_count]
        add_statement = spaces + 'saver_tesra = tf.train.Saver()\n'
        add_statement += spaces + 'saver_tesra.save(%s, "%s")\n' % (self.sess_name, self.checkpoint_path)
        add_statement += spaces + 'print("save model success")\n'
        add_statement += spaces + 'return True\n'
        self.statement_list[self.run_code_count] = run_statement + add_statement
        # 如果run语句在循环内则记住循环列号
        if for_cycle_count_list:
            for _ in range(len(for_cycle_count_list)):
                is_run_in_cycle = True
                for i in range(for_cycle_count_list[0]+1, self.run_code_count+1):
                    if self.spaces_num_list[i] < 0:
                        continue  # 忽视注释行与空行
                    elif self.spaces_num_list[i] <= self.spaces_num_list[for_cycle_count_list[0]]:
                        is_run_in_cycle = False
                        break
                if is_run_in_cycle:
                    count = for_cycle_count_list[0]
                    for_cycle_count_list.append(count)
                for_cycle_count_list.pop(0)
        if while_cycle_count_list:
            for _ in range(len(while_cycle_count_list)):
                is_run_in_cycle = True
                for i in range(while_cycle_count_list[0]+1, self.run_code_count+1):
                    if self.spaces_num_list[i] < 0:
                        continue  # 忽视注释行与空行
                    elif self.spaces_num_list[i] <= self.spaces_num_list[while_cycle_count_list[0]]:
                        is_run_in_cycle = False
                        break
                if is_run_in_cycle:
                    count = while_cycle_count_list[0]
                    step_name = self.step_name_list[0]
                    while_cycle_count_list.append(count)
                    self.step_name_list.append(step_name)
                while_cycle_count_list.pop(0)
                self.step_name_list.pop(0)
        # 更改循环
        if for_cycle_count_list:
            for cycle_count in for_cycle_count_list:
                self.statement_list[cycle_count] = self.statement_list[cycle_count].split('in')[0] + ' in range(1):\n'
        if while_cycle_count_list:
            for cycle_count in while_cycle_count_list:
                step_name = self.step_name_list[while_cycle_count_list.index(cycle_count)]
                step_value = None
                for count in range(cycle_count-1, self.func_count, -1):
                    step_value_list = re.findall(re.compile(r'^\s*%s\s*=\s*(\S+)' % step_name),
                                                 self.statement_list[count].split('#')[0])
                    if len(step_value_list) > 0:
                        step_value = step_value_list[0]
                        break
                if step_value:
                    self.statement_list[cycle_count] = self.statement_list[cycle_count].split('<')[0] +\
                                                       ' <%s+1:\n' % step_value
        # 注释多余启动代码
        start_func_count = None
        for count in range(self.entrance_count+1, len(self.statement_list)):
            start_func_code_list = re.findall(re.compile(r'%s\s*?\(' % self.func_name), self.statement_list[count])
            if len(start_func_code_list) > 0:
                start_func_count = count
                break
        if not start_func_count:
            print('Function not called at file entry')
            exit(1)
        for count in range(start_func_count+1, len(self.statement_list)):
            statement = self.statement_list[count]
            if statement.strip().startswith('#') or statement.strip() == '':
                continue  # 忽略空行和已注释行
            else:
                line_list = statement.split('\n')
                new_line_list = []
                for line in line_list:
                    new_line = '#' + line
                    new_line_list.append(new_line)
                self.statement_list[count] = '\n'.join(new_line_list)
        # 将改好的代码写入文件
        file_data = ''
        for statement in self.statement_list:
            file_data += statement
        with open(self.new_file_name, 'w', encoding="utf-8") as f:
            f.write(file_data)

    def run(self):
        self.single_divide_into_statement()
        self.get_run_message()
        self.write_off_lines()
        a = os.system("python3 save_model.py")
        if a == 0:
            return True
