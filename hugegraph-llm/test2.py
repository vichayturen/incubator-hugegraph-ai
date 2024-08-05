# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


# 测试不同策略的生成的效果差异
# 包括base，basis和two step
# base是原始策略
# basis在原始策略中加入对上下文中没有对应答案的应对方式的提示
# two-step让llm先找出对应的上下文，第二轮会话再问

import json
from typing import Tuple

import pandas as pd
from hugegraph_llm.operators.graph_rag_task import GraphRAG
from hugegraph_llm.models.llms.init_llm import LLMs


test_data = pd.read_excel("/media/vv/8ee675f7-ec97-4bf1-95d5-b326aece4f1e/code/hugegraph/data/事故认定书QueryCase.xlsx", sheet_name="Sheet1")
df = pd.DataFrame(columns=["User Query", "Correct Answer (参考)","Graph Result","Base Answer", "Basis Answer", "Two Step Answer"], dtype=str)


def tow_step_answer(question: str) -> Tuple[str]:
    template1 = (
        "Context information is below.\n"
        "# Context:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "# Query:\n"
        "{query_str}\n"
        "# Instruction:\n"
        "The above 10 context contain 0 or more paragraphs related to query. Please select the paragraph related to Query from the context. If not, please answer."
    )

    template2 = (
        "Please answer the question according to the relevant context: {query_str}"
    )

    template3 = (
        "Context information is below.\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Given the context information and not prior knowledge, answer the query.\n"
        "Note that you need to first find the basis from the above context, and then answer the question based on this basis.\n"
        "If there is no clear basis in the context, you must answer: there is no knowledge related to the question in the knowledge base.\n"
        "Query: {query_str}\n"
        "Answer: "
    )

    # question = "如果驾驶员未按照交通信号灯通行，造成交通事故，应该承担什么责任？"
    llm = LLMs().get_llm()
    context = GraphRAG().extract_keyword().match_keyword_to_id().query_graph_for_rag().merge_dedup_rerank().run(query=question)
    graph_result = context['graph_result']
    prompt1 = template1.format(
        context_str="\n\n".join([f"{i+1}. {text}" for i, text in enumerate(graph_result)]),
        query_str=question
    )
    answer1 = llm.generate(prompt=prompt1)
    prompt2 = template2.format(query_str=question)
    messages = [
        {"role": "user", "content": prompt1},
        {"role": "assistant", "content": answer1},
        {"role": "user", "content": prompt2},
    ]
    answer2 = llm.generate(messages=messages)
    base_answer = GraphRAG().synthesize_answer(
        vector_only_answer=False,
        graph_only_answer=True
    ).run(**context)["graph_only_answer"]
    answer_with_basis = GraphRAG().synthesize_answer(
        prompt_template=template3,
        vector_only_answer=False,
        graph_only_answer=True
    ).run(**context)["graph_only_answer"]
    print("=================================")
    print(prompt1)
    print("=================================")
    print(answer1)
    print("=================================")
    print(prompt2)
    print("=================================")
    print(answer2)
    print("=================================")
    return json.dumps(graph_result, ensure_ascii=False, indent=4), base_answer, answer_with_basis, answer2


if __name__ == '__main__':
    for i in range(10):
        question = test_data.loc[i, "User Query"]
        label = test_data.loc[i, "Correct Answer (参考)"]
        graph_result, base_answer, answer_with_basis, answer2 = tow_step_answer(question)
        df.loc[i] = [question,label, graph_result, base_answer, answer_with_basis, answer2]
    df.to_excel("./rag_result.xlsx", index=False)
