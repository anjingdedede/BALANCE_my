import re
from string import digits
#from src.feature_extraction.predicate_features import *
#from src.feature_extraction.extract_features import *
from src.feature_extraction.extract_features import get_alias2table, pre2seq, get_value_reps_mean

class BagOfOperators(object):
    """
    该类用于从查询计划中提取相关的操作符和值信息。
    它可以处理不同类型的查询计划节点，并将其转换为特定的表示形式。
    """
    def __init__(self):
        """
        初始化 BagOfOperators 类的实例。
        设置替换规则、去除数字的转换表、感兴趣的操作符列表，
        并初始化相关操作符和值的存储变量。
        """
        # 定义替换规则，用于去除字符串中的特定字符
        self.replacings = [(" ", ""), ("(", ""), (")", ""), ("[", ""), ("]", ""), ("::text", "")]
        # 创建一个转换表，用于去除字符串中的数字
        self.remove_digits = str.maketrans("", "", digits)
        # 定义感兴趣的操作符列表
        self.INTERESTING_OPERATORS = [
            "Seq Scan",
            "Hash Join",
            "Nested Loop",
            "CTE Scan",
            "Index Only Scan",
            "Index Scan",
            "Merge Join",
            "Sort",
        ]

        # 初始化相关操作符和值的存储变量
        self.relevant_operators = None
        self.relevant_values = None


    def value_from_plan(self, plan):
        """
        从给定的查询计划中提取相关的值信息。

        参数:
        plan (dict): 表示查询计划的字典。

        返回:
        list: 包含相关值表示的列表。
        """
        # 初始化相关值列表
        self.relevant_values = []
        # 调用解析函数处理计划
        self._parse_value_plan(plan)

        return self.relevant_values

    def _parse_value_plan(self, plan):
        """
        递归解析查询计划，提取相关的值信息。

        参数:
        plan (dict): 表示查询计划的字典。
        """
        # 初始化别名到表名的映射
        alias2table = {}
        # 调用函数填充别名到表名的映射
        get_alias2table(plan, alias2table)

        # 获取当前节点的类型
        node_type = plan["Node Type"]
        # 检查节点类型是否在感兴趣的操作符列表中
        if node_type in self.INTERESTING_OPERATORS:
            # 解析当前节点的值表示
            node_value_representation = self._parse_value_node(plan, alias2table)
            # 将值表示添加到相关值列表中
            self.relevant_values.append(node_value_representation)
        # 检查计划中是否包含子计划
        if "Plans" not in plan:
            return
        # 递归处理每个子计划
        for sub_plan in plan["Plans"]:
            self._parse_value_plan(sub_plan)

    def _parse_value_node(self, node, alias2table):
        """
        解析单个节点，提取其值表示。

        参数:
        node (dict): 表示查询计划节点的字典。
        alias2table (dict): 别名到表名的映射。

        返回:
        object: 节点的值表示。
        """
        # 初始化关系名和索引名
        relation_name, index_name = None, None
        # 检查节点中是否包含关系名
        if 'Relation Name' in node:
            relation_name = node['Relation Name']
        # 检查节点中是否包含索引名
        if 'Index Name' in node:
            index_name = node['Index Name']

        # 初始化节点的值表示
        node_value_representation = None

        # 根据节点类型进行不同的处理
        if node["Node Type"] == "Seq Scan":
            # 检查节点中是否包含过滤条件
            if 'Filter' in node:
                # 将过滤条件转换为序列
                condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
                # 计算过滤条件的平均值表示
                node_value_representation = get_value_reps_mean(condition_seq_filter, relation_name, index_name)
        elif node["Node Type"] == "Index Only Scan":
            # 检查节点中是否包含索引条件
            if 'Index Cond' in node:
                # 将索引条件转换为序列
                condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
                # 计算索引条件的平均值表示
                node_value_representation = get_value_reps_mean(condition_seq_index, relation_name, index_name)
        elif node["Node Type"] == "Index Scan":
            # 检查节点中是否包含过滤条件
            if 'Filter' in node:
                # 将过滤条件转换为序列
                condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
            else:
                # 如果不包含过滤条件，初始化为空列表
                condition_seq_filter = []
            # 检查节点中是否包含索引条件
            if 'Index Cond' in node:
                # 将索引条件转换为序列
                condition_seq_index = pre2seq(node['Index Cond'], alias2table, relation_name, index_name)
            else:
                # 如果不包含索引条件，初始化为空列表
                condition_seq_index = []
            # 计算过滤条件和索引条件的平均值表示
            node_value_representation = get_value_reps_mean(condition_seq_filter+condition_seq_index, relation_name, index_name)
        elif node["Node Type"] == "CTE Scan":
            # 获取 CTE 名称作为关系名
            relation_name = node['CTE Name']
            # 检查节点中是否包含过滤条件且父关系不是 Inner
            if 'Filter' in node and node['Parent Relationship'] != 'Inner':
                # 将过滤条件转换为序列
                condition_seq_filter = pre2seq(node['Filter'], alias2table, relation_name, index_name)
                # 计算过滤条件的平均值表示
                node_value_representation = get_value_reps_mean(condition_seq_filter, relation_name, index_name)
        return node_value_representation

    def boo_from_plan(self, plan):
        """
        从给定的查询计划中提取相关的操作符信息。

        参数:
        plan (dict): 表示查询计划的字典。

        返回:
        list: 包含相关操作符表示的列表。
        """
        # 初始化相关操作符列表
        self.relevant_operators = []
        # 调用解析函数处理计划
        self._parse_plan(plan)

        return self.relevant_operators

    def _parse_plan(self, plan):
        """
        递归解析查询计划，提取相关的操作符信息。

        参数:
        plan (dict): 表示查询计划的字典。
        """
        # 获取当前节点的类型
        node_type = plan["Node Type"]

        # 检查节点类型是否在感兴趣的操作符列表中
        if node_type in self.INTERESTING_OPERATORS:
            # 解析当前节点的操作符表示
            node_representation = self._parse_node(plan)
            # 将操作符表示添加到相关操作符列表中
            self.relevant_operators.append(node_representation)
        # 检查计划中是否包含子计划
        if "Plans" not in plan:
            return
        # 递归处理每个子计划
        for sub_plan in plan["Plans"]:
            self._parse_plan(sub_plan)

    def _stringify_attribute_columns(self, node, attribute):
        """
        将节点的属性值转换为字符串表示，去除特定字符和数字。

        参数:
        node (dict): 表示查询计划节点的字典。
        attribute (str): 要处理的属性名。

        返回:
        str: 属性的字符串表示。
        """
        # 初始化属性表示
        attribute_representation = f"{attribute.replace(' ', '')}_"
        # 检查节点中是否包含该属性
        if attribute not in node:
            return attribute_representation

        # 获取属性值
        value = node[attribute]

        # 应用替换规则
        for replacee, replacement in self.replacings:
            value = value.replace(replacee, replacement)

        # 去除双引号和单引号内的内容
        value = re.sub('".*?"', "", value)
        value = re.sub("'.*?'", "", value)
        # 去除数字
        value = value.translate(self.remove_digits)

        return value

    def _stringify_list_attribute(self, node, attribute):
        """
        将节点的列表属性值转换为字符串表示。

        参数:
        node (dict): 表示查询计划节点的字典。
        attribute (str): 要处理的属性名。

        返回:
        str: 属性的字符串表示。
        """
        # 初始化属性表示
        attribute_representation = f"{attribute.replace(' ', '')}_"
        # 检查节点中是否包含该属性
        if attribute not in node:
            return attribute_representation

        # 确保属性值是列表类型
        assert isinstance(node[attribute], list)
        # 获取属性值
        value = node[attribute]

        # 遍历列表元素，拼接属性表示
        for element in value:
            attribute_representation += f"{element}_"

        return attribute_representation

    def _parse_bool_attribute(self, node, attribute):
        """
        将节点的布尔属性值转换为字符串表示。

        参数:
        node (dict): 表示查询计划节点的字典。
        attribute (str): 要处理的属性名。

        返回:
        str: 属性的字符串表示。
        """
        # 初始化属性表示
        attribute_representation = f"{attribute.replace(' ', '')}_"

        # 检查节点中是否包含该属性
        if attribute not in node:
            return attribute_representation

        # 获取属性值
        value = node[attribute]
        # 拼接属性值到属性表示
        attribute_representation += f"{value}_"

        return attribute_representation

    def _parse_string_attribute(self, node, attribute):
        """
        将节点的字符串属性值转换为字符串表示。

        参数:
        node (dict): 表示查询计划节点的字典。
        attribute (str): 要处理的属性名。

        返回:
        str: 属性的字符串表示。
        """
        # 初始化属性表示
        attribute_representation = f"{attribute.replace(' ', '')}_"

        # 检查节点中是否包含该属性
        if attribute not in node:
            return attribute_representation

        # 获取属性值
        value = node[attribute]
        # 拼接属性值到属性表示
        attribute_representation += f"{value}_"

        return attribute_representation

    def _parse_seq_scan(self, node):
        """
        解析 Seq Scan 节点，生成其字符串表示。

        参数:
        node (dict): 表示 Seq Scan 节点的字典。

        返回:
        str: Seq Scan 节点的字符串表示。
        """
        # 确保节点包含关系名
        assert "Relation Name" in node

        # 初始化节点表示
        node_representation = ""
        # 添加关系名到节点表示
        node_representation += f"{node['Relation Name']}_"

        # 添加过滤条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Filter")

        return node_representation

    def _parse_index_scan(self, node):
        """
        解析 Index Scan 节点，生成其字符串表示。

        参数:
        node (dict): 表示 Index Scan 节点的字典。

        返回:
        str: Index Scan 节点的字符串表示。
        """
        # 确保节点包含关系名
        assert "Relation Name" in node

        # 初始化节点表示
        node_representation = ""
        # 添加关系名到节点表示
        node_representation += f"{node['Relation Name']}_"

        # 添加过滤条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Filter")
        # 添加索引条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Index Cond")

        return node_representation

    def _parse_index_only_scan(self, node):
        """
        解析 Index Only Scan 节点，生成其字符串表示。

        参数:
        node (dict): 表示 Index Only Scan 节点的字典。

        返回:
        str: Index Only Scan 节点的字符串表示。
        """
        # 确保节点包含关系名
        assert "Relation Name" in node

        # 初始化节点表示
        node_representation = ""
        # 添加关系名到节点表示
        node_representation += f"{node['Relation Name']}_"

        # 添加索引条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Index Cond")

        return node_representation

    def _parse_cte_scan(self, node):
        """
        解析 CTE Scan 节点，生成其字符串表示。

        参数:
        node (dict): 表示 CTE Scan 节点的字典。

        返回:
        str: CTE Scan 节点的字符串表示。
        """
        # 确保节点包含 CTE 名称
        assert "CTE Name" in node

        # 初始化节点表示
        node_representation = ""
        # 添加 CTE 名称到节点表示
        node_representation += f"{node['CTE Name']}_"

        # 添加过滤条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Filter")

        return node_representation

    def _parse_nested_loop(self, node):
        """
        解析 Nested Loop 节点，生成其字符串表示。

        参数:
        node (dict): 表示 Nested Loop 节点的字典。

        返回:
        str: Nested Loop 节点的字符串表示。
        """
        # 初始化节点表示
        node_representation = ""

        # 添加连接过滤条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Join Filter")

        return node_representation

    def _parse_hash_join(self, node):
        """
        解析 Hash Join 节点，生成其字符串表示。

        参数:
        node (dict): 表示 Hash Join 节点的字典。

        返回:
        str: Hash Join 节点的字符串表示。
        """
        # 初始化节点表示
        node_representation = ""

        # 添加连接过滤条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Join Filter")
        # 添加哈希条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Hash Cond")

        return node_representation

    def _parse_merge_join(self, node):
        """
        解析 Merge Join 节点，生成其字符串表示。

        参数:
        node (dict): 表示 Merge Join 节点的字典。

        返回:
        str: Merge Join 节点的字符串表示。
        """
        # 初始化节点表示
        node_representation = ""

        # 添加合并条件的字符串表示到节点表示
        node_representation += self._stringify_attribute_columns(node, "Merge Cond")

        return node_representation

    def _parse_sort(self, node):
        """
        解析 Sort 节点，生成其字符串表示。

        参数:
        node (dict): 表示 Sort 节点的字典。

        返回:
        str: Sort 节点的字符串表示。
        """
        # 初始化节点表示
        node_representation = ""

        # 添加排序键的字符串表示到节点表示
        node_representation += self._stringify_list_attribute(node, "Sort Key")

        return node_representation

    def _parse_node(self, node):
        """
        解析单个节点，根据节点类型生成其字符串表示。

        参数:
        node (dict): 表示查询计划节点的字典。

        返回:
        str: 节点的字符串表示。
        """
        # 初始化节点表示，包含节点类型
        node_representation = f"{node['Node Type'].replace(' ', '')}_"

        # 根据节点类型进行不同的处理
        if node["Node Type"] == "Seq Scan":
            # 添加 Seq Scan 节点的字符串表示
            node_representation += f"{self._parse_seq_scan(node)}"
        elif node["Node Type"] == "Index Only Scan":
            # 添加 Index Only Scan 节点的字符串表示
            node_representation += f"{self._parse_index_only_scan(node)}"
        elif node["Node Type"] == "Index Scan":
            # 添加 Index Scan 节点的字符串表示
            node_representation += f"{self._parse_index_scan(node)}"
        elif node["Node Type"] == "CTE Scan":
            # 添加 CTE Scan 节点的字符串表示
            node_representation += f"{self._parse_cte_scan(node)}"
        elif node["Node Type"] == "Nested Loop":
            # 添加 Nested Loop 节点的字符串表示
            node_representation += f"{self._parse_nested_loop(node)}"
        elif node["Node Type"] == "Hash Join":
            # 添加 Hash Join 节点的字符串表示
            node_representation += f"{self._parse_hash_join(node)}"
        elif node["Node Type"] == "Merge Join":
            # 添加 Merge Join 节点的字符串表示
            node_representation += f"{self._parse_merge_join(node)}"
        elif node["Node Type"] == "Sort":
            # 添加 Sort 节点的字符串表示
            node_representation += f"{self._parse_sort(node)}"
        else:
            # 抛出异常，如果节点类型不支持
            raise ValueError("_parse_node called with unsupported Node Type.")

        return node_representation
