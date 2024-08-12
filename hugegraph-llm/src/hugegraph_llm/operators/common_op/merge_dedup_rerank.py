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


from typing import Dict, Any, List, Literal

import jieba
from hugegraph_llm.models.embeddings.base import BaseEmbedding
from nltk.translate.bleu_score import sentence_bleu


def get_score(query: str, content: str) -> float:
    query_tokens = jieba.lcut(query)
    content_tokens = jieba.lcut(content)
    return sentence_bleu([query_tokens], content_tokens)


class MergeDedupRerank:
    def __init__(
            self,
            embedding: BaseEmbedding,
            topk: int = 10,
            policy: Literal["bleu", "priority"] = "bleu"
    ):
        self.embedding = embedding
        self.topk = topk
        if policy == "bleu":
            self.rerank_func = self._bleu_rerank
        elif policy == "priority":
            self.rerank_func = self._priority_rerank
        else:
            raise ValueError(f"Unimplemented policy {policy}.")

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        query = context.get("query")

        vector_result = context.get("vector_result", [])
        vector_result = self.rerank_func(query, vector_result)[:self.topk]

        graph_result = context.get("graph_result", [])
        graph_result = self.rerank_func(query, graph_result)[:self.topk]

        context["vector_result"] = vector_result
        context["graph_result"] = graph_result

        return context

    def _bleu_rerank(self, query: str, results: List[str]):
        results = list(set(results))
        result_score_list = [[res, get_score(query, res)] for res in results]
        result_score_list.sort(key=lambda x: x[1], reverse=True)
        return [res[0] for res in result_score_list]

    def _priority_rerank(self, query: str, results: List[str]):
        # TODO: 逻辑应该是：
        # 1. 精准召回 > 模糊召回
        # 2. 一度邻居 > 二度邻居（相关度更高）
        # 3. 某类点的优先级高于其他，比如 Law(法条) 高于车/人/地点
        results = list(set(results))
        result_score_list = [[res, get_score(query, res)] for res in results]
        result_score_list.sort(key=lambda x: x[1], reverse=True)
        return [res[0] for res in result_score_list]
