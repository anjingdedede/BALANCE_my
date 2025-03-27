from src.feature_extraction.database_loader import *
from src.feature_extraction.plan_features import *
from src.feature_extraction.sample_bitmap import *


def add_sample_bitmap(input_path, output_path, data, sample, sample_num):
    with open(input_path, 'r') as ff:
        with open(output_path, 'w') as f:
            for count, plan in enumerate(ff.readlines()):
                print (count)
                parsed_plan = json.loads(plan)
                nodes_with_sample = []
                for node in parsed_plan['seq']:
                    bitmap_filter = []
                    bitmap_index = []
                    bitmap_other = []
                    if node != None and 'condition' in node:
                        predicates = node['condition']
                        if len(predicates) > 0:
                            root = TreeNode(predicates[0], None)
                            if len(predicates) > 1:
                                recover_tree(predicates[1:], root)
                            bitmap_other = get_bitmap(root, data, sample, sample_num)
                    if node != None and 'condition_filter' in node:
                        predicates = node['condition_filter']
                        if len(predicates) > 0:
                            root = TreeNode(predicates[0], None)
                            if len(predicates) > 1:
                                recover_tree(predicates[1:], root)
                            bitmap_filter = get_bitmap(root, data, sample, sample_num)
                    if node != None and 'condition_index' in node:
                        predicates = node['condition_index']
                        if len(predicates) > 0:
                            root = TreeNode(predicates[0], None)
                            if len(predicates) > 1:
                                recover_tree(predicates[1:], root)
                            bitmap_index = get_bitmap(root, data, sample, sample_num)
                    if len(bitmap_filter) > 0 or len(bitmap_index) > 0 or len(bitmap_other) > 0:
                        bitmap = [1 for _ in range(sample_num)]
                        bitmap = bitand(bitmap, bitmap_filter)
                        bitmap = bitand(bitmap, bitmap_index)
                        bitmap = bitand(bitmap, bitmap_other)
                        node['bitmap'] = ''.join([str(x) for x in bitmap])
                    nodes_with_sample.append(node)
                parsed_plan['seq'] = nodes_with_sample
                f.write(json.dumps(parsed_plan))
                f.write('\n')


def get_subplan(root):
    results = []
    if 'Actual Rows' in root and 'Actual Total Time' in root and 'Actual Rows' in root > 0:
        results.append((root, root['Actual Total Time'], root['Actual Rows']))
    if 'Plans' in root:
        for plan in root['Plans']:
            results += get_subplan(plan)
    return results

def get_plan(root):
    # return (root, root['Actual Total Time'], root['Actual Rows'])
    return (root, 0, 0)

class PlanInSeq(object):
    def __init__(self, seq, cost, cardinality):
        self.seq = seq
        self.cost = cost
        self.cardinality = cardinality

def get_alias2table(root, alias2table):
    """
    递归遍历计划树，将表别名映射到实际表名，并存储在 alias2table 字典中。

    参数:
    root (dict): 计划树的根节点，通常是一个包含计划信息的字典。
    alias2table (dict): 用于存储表别名到实际表名映射的字典。
    """
    # 检查当前节点是否包含 'Relation Name' 和 'Alias' 字段
    if 'Relation Name' in root and 'Alias' in root:
        # 如果包含，则将别名映射到实际表名
        alias2table[root['Alias']] = root['Relation Name']
    # 检查当前节点是否包含子计划
    if 'Plans' in root:
        # 遍历所有子计划
        for child in root['Plans']:
            # 递归调用自身处理子计划
            get_alias2table(child, alias2table)





def feature_extractor(plans):
    with open('plans2seq.json', 'w') as out:
        plans_seq = []
        for plan in plans:
            if plan['Node Type'] == 'Aggregate':
                plan = plan['Plans'][0]

            alias2table = {}
            get_alias2table(plan, alias2table)
            plan_seq = []
            plan2seq_dfs(plan, alias2table, plan_seq)
            plans_seq.append(plan_seq)
            out.write(class2json(plan_seq) + '\n')

    return plans_seq