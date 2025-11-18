import json
import logging
import pdb
import time
from typing import List, Dict, Tuple, Any
from src.models.base_rag import BaseRAG
from src.utils.utils import get_response_with_retry, fix_json_response
from colorama import Fore, Style, init
from queue import Queue

# Initialize colorama
init()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


class Node:
    def __init__(self, level, question, rationale="", score=0.5):
        self.level = level
        self.question = question
        self.rationale = rationale
        self.score = score

    def is_endnode(self):
        return not (self.question or self.rationale or self.score)

class LogicRAG(BaseRAG):
    
    def __init__(self, corpus_path: str = None, cache_dir: str = "./cache", filter_repeats: bool = False):
        """Initialize the LogicRAG system."""
        super().__init__(corpus_path, cache_dir)
        self.max_rounds = 3  # Default max rounds for iterative retrieval
        self.MODEL_NAME = "LogicRAG"
        self.filter_repeats = filter_repeats  # Option to filter repeated chunks across rounds
    
    def set_max_rounds(self, max_rounds: int):
        """Set the maximum number of retrieval rounds."""
        self.max_rounds = max_rounds
    
    def refine_summary_with_context(self, question: str, new_contexts: List[str], 
                                  current_summary: str = "") -> str:
        """
        Generate a new summary or refine an existing one based on newly retrieved contexts.
        
        Args:
            question: The original question
            new_contexts: Newly retrieved context chunks
            current_summary: Current information summary (if any)
            
        Returns:
            A concise summary of all relevant information so far
        """
        try:
            context_text = "\n".join(new_contexts)
            
            if not current_summary:
                # Generate initial summary
                prompt = f"""Please create a concise summary of the following information as it relates to answering this question:

Question: {question}

Information:
{context_text}

Your summary should:
1. Include all relevant facts that might help answer the question
2. Exclude irrelevant information
3. Be clear and concise
4. Preserve specific details, dates, numbers, and names that may be relevant

Summary:"""
            else:
                # Refine existing summary with new information
                prompt = f"""Please refine the following information summary using newly retrieved information.

Question: {question}

Current summary:
{current_summary}

New information:
{context_text}

Your refined summary should:
1. Integrate new relevant facts with the existing summary
2. Remove redundancies
3. Remain concise while preserving all important information
4. Prioritize information that helps answer the question
5. Maintain specific details, dates, numbers, and names that may be relevant

Refined summary:"""
            
            summary = get_response_with_retry(prompt)
            return summary
            
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating/refining summary: {e}{Style.RESET_ALL}")
            # If error occurs, concatenate current summary with new contexts as fallback
            if current_summary:
                return f"{current_summary}\n\nNew information:\n{context_text}"
            return context_text
    
    def warm_up_analysis(self, question: str, info_summary: str) -> Dict:
        """
        This is a warm-up analysis, which is used to analyze if the question can be answered with simple fact retrieval, without any dependency analysis.
        
        Args:
            question: The original question
            info_summary: Current information summary
            
        Returns:
            Dictionary with analysis results
        """
        try:
            prompt = f"""Question: {question}

Available Information:
{info_summary}

Based on the information provided, please analyze:
1. Can the question be answered completely with this information? (Yes/No)
2. What specific information is missing, if any?
3. What specific question should we ask to find the missing information?
4. Summarize our current understanding based on available information.
5. What are the key dependencies needed to answer this question?
6. Why is information missing? (max 20 words)

Please format your response as a JSON object with these keys:
- "can_answer": boolean
- "missing_info": string
- "subquery": string
- "current_understanding": string
- "dependencies": list of strings (key information dependencies)
- "missing_reason": string (brief explanation why info is missing, max 20 words)"""
            
            response = get_response_with_retry(prompt)
            
            # Clean up response to ensure it's valid JSON
            response = response.strip()
            
            # Remove any markdown code block markers
            response = response.replace('```json', '').replace('```', '')
            
            # Parse the cleaned response using fix_json_response
            result = fix_json_response(response)
            if result is None:
                return {
                    "can_answer": True,
                    "missing_info": "",
                    "subquery": question,
                    "current_understanding": "Failed to parse reflection response.",
                    "dependencies": ["Information relevant to the question"],
                    "missing_reason": "Parse error occurred"
                }
            
            # Validate required fields
            required_fields = ["can_answer", "missing_info", "subquery", "current_understanding"]
            if not all(field in result for field in required_fields):
                logger.error(f"{Fore.RED}Missing required fields in response: {response}{Style.RESET_ALL}")
                raise ValueError("Missing required fields")
            
            # Add default values for new interpretability fields if missing
            if "dependencies" not in result:
                result["dependencies"] = ["Information relevant to the question"]
            if "missing_reason" not in result:
                result["missing_reason"] = "Additional context needed" if not result["can_answer"] else "No missing information"
            
            # Ensure boolean type for can_answer
            result["can_answer"] = bool(result["can_answer"])
            
            # Ensure non-empty subquery
            if not result["subquery"]:
                result["subquery"] = question
            
            return result
                
        except Exception as e:
            logger.error(f"{Fore.RED}Error in analyze_dependency_graph: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "missing_info": "",
                "subquery": question,
                "current_understanding": f"Error during analysis: {str(e)}",
                "dependencies": ["Information relevant to the question"],
                "missing_reason": "Analysis error occurred"
            }

    def dependency_aware_rag(self, question: str, info_summary: str, dependencies: List[str], idx: int) -> str:
        """
        similar to "self.analyze_dependency_graph" that analyzes whether the current information summary is sufficient to answer the question,
        this function analyzes whether the current information summary is sufficient to answer the question with the decomposed dependencies as references.

        And the function will answer whether the question can be answered, and if not, it will update the current query with dependencies as references.

        Args:
            question: str
            info_summary: str
            dependencies: List[str]
            idx: int
        """

        try:
            prompt = f"""
            We pre-parsed the question into a list of dependencies, and the dependencies are sorted in a topological order, below is the question, the information summary, and the decomposed dependencies:

            Question: {question}

            Available Information:
            {info_summary}

            Decomposed dependencies:
            {dependencies}

            Current dependency to be answered:
            {dependencies[idx]}

            Please analyze the question and the information summary, and the decomposed dependencies, and answer the following questions:
            Please analyze:
            1. Can the question be answered completely with this information? (Yes/No)
            2. Summarize our current understanding based on available information.

            Please format your response as a JSON object with these keys:
            - "can_answer": boolean
            - "current_understanding": string
            """
            response = get_response_with_retry(prompt)
            result = fix_json_response(response)
            return result
        except Exception as e:
            logger.error(f"{Fore.RED}Error in dependency_aware_rag: {e}{Style.RESET_ALL}")
            return {
                "can_answer": True,
                "current_understanding": f"Error during analysis: {str(e)}",
            }

    def generate_answer(self, question: str, info_summary: str) -> str:
        """Generate final answer based on the information summary."""
        try:
            prompt = f"""You must give ONLY the direct answer in the most concise way possible. DO NOT explain or provide any additional context.
If the answer is a simple yes/no, just say "Yes." or "No."
If the answer is a name, just give the name.
If the answer is a date, just give the date.
If the answer is a number, just give the number.
If the answer requires a brief phrase, make it as concise as possible.

Question: {question}

Information Summary:
{info_summary}

Remember: Be concise - give ONLY the essential answer, nothing more.
Ans: """
            
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""
        
        
    def generate_cot_track(self, question: str, info_summary: str) -> str:
        """Generate final answer based on the information summary."""
        try:
            prompt = f"""Answer the following question based on the provided context and provide a detailed reasoning process.\n

Question: {question}

provided context:
{info_summary}

Provide your reasoning process: """

            return get_response_with_retry(prompt.format(
                context=" ".join(info_summary),
                question=question
            ))
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""
        
    def check_score(self, question: str, info_summary: str) -> str:
        """Generate final answer based on the information summary."""
        try:
            prompt = f"""Evaluate whether the given rationale actually answers the question.
    Scoring rules (0–10):
    - 10: Fully and directly answers the question, clearly grounded in the article’s content.
    - 7–9: Answers the question, mostly grounded in the article; minor gaps or implicit evidence.
    - 4–6: Partially answers or is vague/hedged; grounding is weak or mixed.
    - 1–3: Does **not** answer the question, or states/assumes the information is unavailable in the article, or merely summarizes/restates without a conclusion, or relies on external/unsupported claims.
    - 0: Entirely unrelated or contradicts the article.
    Always judge answerability from the article alone. If the article lacks the needed information to answer, assign ≤3 regardless of reasoning quality.
Question: {question}

the given rationale:
{info_summary}

You should only output a single integer between 0 and 10 (inclusive). No other text. """
            
            return int(get_response_with_retry(prompt))/ 10.0
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""       

    def extract(self, question: str, info_summary: str, node_memory) -> str:
        """Generate final answer based on the information summary."""
        try:
            node_map = {}
            for node in node_memory:
                node_map[node.question] = node.rationale
            prompt = f"""Please extract questions from the following list that might use the answer of q to update their questions. Question list:\n"""
            idx = 1
            for node in node_memory:
                prompt += (f"{idx}: " + node.question + "\n")
                idx += 1
            prompt += ("q: " + question + "\n" + "The answer of q: " + info_summary + "\n")
            prompt += "Please output the indices of the extracted questions as a string in list format, where each element represents the index of a question, in the form of '[\"1\", \"2\", \"3\"]' Do not add explanations, code blocks, or any other content."            
            prompt += "Your extracted questions:\n"
            string_question = get_response_with_retry(prompt)
            list_question = json.loads(string_question)
            memory_list = []
            for ids in list_question:
                idx = int(ids)
                memory_list.append(node_memory[idx-1])
            return memory_list         
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return "" 
        
        
    def update(self, question: str, info_summary: str,prev_question: str, prev_rationale: str ) -> str:
        """Generate final answer based on the information summary."""
        try:
            head = f"""Please update the reasoning to question b based on the reasoning to question a..\n"""

            prompt = head  + "Question a: " + question + "\n" + "The reasoning to question a: " + info_summary + "\n" + "Question b: " + prev_question + "\n" + "The reasoning to question b: " + prev_rationale + "\n" +"Updated reasoning:"
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return "" 
        
    def Rewrite(self, question: str, info_summary: str,prev_question: str ) -> str:
        """Generate final answer based on the information summary."""
        try:
            head = f"""Please rewrite question b based on the answer to question a and replace pronouns or references with the actual entities they refer to.\n"""
            instruction ='''The rewritten question b is:\n'''
            prompt = head + "Question a: " + question + "\n" + "The answer to question a: " + info_summary + "\n" + "Question b: " + prev_question + "\n" + instruction
            return get_response_with_retry(prompt)
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""
    
    
    def Decompose_new(self, question: str ) -> str:
        """Generate final answer based on the information summary."""
        try:
            head = """You are a helpful AI assistant that helps break down questions into minimal necessary sub-questions.    
        Guidelines:
        1. Only break down the question if it requires finding and connecting multiple distinct pieces of information
        2. Each sub-question should target a specific, essential piece of information
        3. Avoid generating redundant or overlapping sub-questions
        4. For questions about impact/significance, focus on:
           - What was the thing/event
           - What was its impact/significance
        5. For comparison questions between two items (A vs B):
           - First identify the specific attribute being compared for each item
           - Then ask about that attribute for each item separately
           - For complex comparisons, add a final question to compare the findings
        6. Logical Progression:
           - Parallel: independent sub-questions that both contribute to answering the original question.
             Example:
             Original: "What are the causes and consequences of climate change on global ecosystems?"
             Output: ["What are the main causes of climate change?", "What are the major consequences of climate change on global ecosystems?"]
           - Sequential: sub-questions that build upon each other step-by-step.
             Example:
             Original: "What university, founded in 1890, is known for its groundbreaking work in economics?"
             Output: ["Which universities were founded in 1890?", "Which of these universities is known for its groundbreaking work in economics?"]
           - Comparative:
             Example 1:
             Original: "Which film has the director who was born earlier, The Secret Invasion or The House Of The Seven Hawks?"
             Output: ["Who directed The Secret Invasion and when was this director born?",
                      "Who directed The House Of The Seven Hawks and when was this director born?"]
             Example 2:
             Original: "Do both films The Reincarnation Of Golden Lotus and I'll Get By (Film) have directors from the same country?"
             Output: ["Who directed The Reincarnation Of Golden Lotus and which country is he/she from?",
                      "Who directed I'll Get By (Film) and which country is he/she from?"]

        7. Keep the total number of sub-questions minimal (usually 2 at most)

        Output format must be a JSON array of strings.
        Example:
        Original: "Were the wireless earbuds Apple introduced in 2016 revolutionary for the market?"
        Output: ["What wireless earbuds did Apple introduce in 2016?",
                 "How did these earbuds impact the wireless earbud market?"]

        Remember: Each sub-question must be necessary and distinct. Do not create redundant questions.
    """
            instruction ="""Now generate the decomposed sub-questions for Q according to the above rules:
    """
            prompt = head  + "The given question Q: " + question + "\n" + instruction + "Please output the decomposed sub-questions as a string in list format, where each element represents the text of a sub-question, in the form of '[\"subq1\", \"subq2\", \"subq3\"]'."
            string_data = get_response_with_retry(prompt)
            try:
              list_data = json.loads(string_data)
            except (json.JSONDecodeError, TypeError) as e:
              list_data = [question]
            return list_data
        except Exception as e:
            logger.error(f"{Fore.RED}Error generating answer: {e}{Style.RESET_ALL}")
            return ""    
    
    
    
    def force_answer(self, sub_question: str, context_sentences: str) -> str:
        """
        Generate a concise answer to a sub-question based on provided context.
        
        Args:
            sub_question: Question to answer
            context_sentences: Context information for answering
            
        Returns:
            Generated answer as a string
        """
        prompt = """Based on the given reasoning process, you must provide an answer with the fewest words to the question.
Only give me the answer and do not output any other words.
        Question: {question}
        Reasoning process: {context}

       Provide the most possible answer:"""
      
        try:
            response = get_response_with_retry(prompt.format(
                context=" ".join(context_sentences),
                question=sub_question
            ))
            return response.strip()
            
        except Exception as e:
            print(f"Error forcing answer: {str(e)}")
            return "Unable to provide an answer due to error"
    
    
    
        
    def _sort_dependencies(self, dependencies: List[str], query) -> List[Tuple]:
        """
        given a list of dependencies and the original query,
        sort the dependencies in a topological order, that is solving a dependency A relies on the solution of the dependent dependency B,
        then B should be before A in the sorted string.

        Args: = 
            dependencies: List[str]
            query: str

            
        For example, if the question is "What is the mayor of the capital of France?",
        the input dependencies for this question are:
        - The capital of France
        - The mayor of this capital

        Then the output should be:
        - The capital of France
        - The mayor of this capital

        there are two steps to solve this problem:
        1. generate the dependency pairs that dependency A relies on dependency B
        2. use graph-based algorithm to sort the dependencies in a topological order

        For example, answering the question "What is the mayor of the capital of France?"
        the input dependencies are:
        - The capital of France
        - The mayor of this capital

        Then the dependency pairs are:
        - [(1, 0)]
        because the mayor of the capital of France relies on the capital of France

        Then the topological order is computed by the self._topological_sort function, which is a graph-based algorithm. The output is a list of indices of the dependencies in the topological order.
        In this case, the output is:
        [0, 1]

        The sorted dependencies are thus:
        - The capital of France
        - The mayor of this capital
        """


        # Step 1: generate the dependency pairs by prompting LLMs
        prompt = f"""
        Given the question:
        Question: {query}

        and its decomposed dependencies:
        Dependencies: {dependencies}

        Please output the dependency pairs that dependency A relies on dependency B, if any. If no dependency pairs are found, output an empty list.

        format your response as a JSON object with these keys:
        - "dependency_pairs": list of tuples of integers
        """
        response = get_response_with_retry(prompt)
        result = fix_json_response(response)
        dependency_pairs = result["dependency_pairs"]

        # Step 2: use graph-based algorithm to sort the dependencies in a topological order
        sorted_dependencies = self._topological_sort(dependencies, dependency_pairs)
        return sorted_dependencies

    @staticmethod
    def _topological_sort(dependencies: List[str], dependencies_pairs: List[Tuple[int, int]]) -> List[str]:
        """
        Use graph-based algorithm to sort the dependencies in a topological order.
        Args:
            dependencies: List[str]
            dependencies_pairs: List[Tuple[int, int]]
        Returns:
            List[str]
        """
        graph = {dep: [] for dep in dependencies}
        
        for dependent_idx, dependency_idx in dependencies_pairs:
            if dependent_idx < len(dependencies) and dependency_idx < len(dependencies):
                dependent = dependencies[dependent_idx]
                dependency = dependencies[dependency_idx]
                graph[dependency].append(dependent)  # dependency -> dependent
        
        visited = set()
        stack = []
        
        def dfs(node):
            if node in visited:
                return
            visited.add(node)
            for neighbor in graph[node]:
                dfs(neighbor)
            stack.append(node)

        for node in graph:
            if node not in visited:
                dfs(node)
        
        return stack[::-1]

    def _retrieve_with_filter(self, query: str, retrieved_chunks_set: set) -> list:
        """
        Retrieve top_k unique chunks not in retrieved_chunks_set. If not enough unique chunks, return as many as possible.
        """
        all_results = self.retrieve(query)
        unique_results = []
        idx = self.top_k
        # If not enough unique in top_k, keep expanding
        while len(unique_results) < self.top_k and idx <= len(self.corpus):
            # Expand retrieval window
            all_results = self.retrieve(query) if idx == self.top_k else self._retrieve_top_n(query, idx)
            unique_results = [chunk for chunk in all_results if chunk not in retrieved_chunks_set]
            idx += self.top_k
        return unique_results[:self.top_k]

    def _retrieve_top_n(self, query: str, n: int) -> list:
        """Retrieve top-n results for a query (helper for filtering)."""
        # Temporarily override top_k
        old_top_k = self.top_k
        self.top_k = n
        results = self.retrieve(query)
        self.top_k = old_top_k
        return results

    def answer_question(self, question: str) -> Tuple[str, List[str], int]:

        info_summary = "" 
        round_count = 0
        current_query = question
        retrieval_history = []
        last_contexts = []  
        dependency_analysis_history = []  
        retrieved_chunks_set = set() if self.filter_repeats else None  # Track retrieved chunks if filtering
        
        print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")
        
        #===============================================
        #== Stage 1: warm up retrieval ==
        if self.filter_repeats:
            new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
            for chunk in new_contexts:
                retrieved_chunks_set.add(chunk)
        else:
            new_contexts = self.retrieve(question)
        last_contexts = new_contexts  
        info_summary = self.refine_summary_with_context(
            question, 
            new_contexts, 
            info_summary
        )

        analysis = self.warm_up_analysis(question, info_summary)

        if analysis["can_answer"]:
            # In this case, the question can be answered with simple fact retrieval, without any dependency analysis
            print(f"Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
            answer = self.generate_answer(question, info_summary)
            # Reset dependency analysis history for simple questions
            self.last_dependency_analysis = []
            return answer, last_contexts, round_count
        else:
            logger.info(f"Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
            logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")

            # sort the dependencies, by first constructing the dependency graphs, then use topological sort to get the sorted dependencies
            sorted_dependencies = self._sort_dependencies(analysis["dependencies"], question)
            dependency_analysis_history.append({"sorted_dependencies": sorted_dependencies})
            logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")
        #===============================================
        #== Stage 2: agentic iterative retrieval ==
        idx = 0 # used to track the current dependency index

        while round_count < self.max_rounds and idx < len(sorted_dependencies):
            round_count += 1
            
            current_query = sorted_dependencies[idx]
            if self.filter_repeats:
                new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
                for chunk in new_contexts:
                    retrieved_chunks_set.add(chunk)
            else:
                new_contexts = self.retrieve(current_query)
            last_contexts = new_contexts  # Save current contexts
            
            
            # Generate or refine information summary with new contexts
            info_summary = self.refine_summary_with_context(
                question, 
                new_contexts, 
                info_summary
            )
            
            logger.info(f"Agentic retrieval at round {round_count}")
            logger.info(f"current query: {current_query}")
            
            analysis = self.dependency_aware_rag(question, info_summary, sorted_dependencies, idx)

            retrieval_history.append({
                "round": round_count,
                "query": current_query,
                "contexts": new_contexts,
            }) 

            dependency_analysis_history.append({
                "round": round_count,
                "query": current_query,
                "analysis": analysis
            })

            if analysis["can_answer"]:
                # Generate and return final answer
                answer = self.generate_answer(question, info_summary)
                # Store dependency analysis history for evaluation access
                self.last_dependency_analysis = dependency_analysis_history
                # We return the last retrieved contexts for evaluation purposes
                return answer, last_contexts, round_count
            else:
                idx += 1
        
        # If max rounds reached, generate best possible answer
        logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        answer = self.generate_answer(question, info_summary)
        # Store dependency analysis history for evaluation access
        self.last_dependency_analysis = dependency_analysis_history
        return answer, last_contexts, round_count

    def answer_question_dynamic(self, question: str) -> Tuple[str, List[str], int]:
    
        EPSILON2 = 0.7
        MAX_DEPTH = 3
        MAX_BRANCH = 3
        FLAG = 0
        
        root = Node(level=0, question=question) 
        N = Queue()
        N.put(root)
        node_memory = [root]
        while not N.empty():
            nt = N.get()  #取出最前面的元素
            if nt.is_endnode():  
                continue
            if nt.level > MAX_DEPTH:  #level表示节点的深度，如果深度超过最大深度则跳出循环为3
                break           
            new_contexts = self.retrieve(nt.question)
            # context_length_total += len(new_contexts)
            r1 = self.generate_cot_track(nt.question, new_contexts)
            nt.rationale = r1
            nt.score = self.check_score(nt.question, nt.rationale)
            print("question:", nt.question)
            print("rationale:", nt.rationale)
            print("score", nt.score)
            if nt.level > MAX_DEPTH:  #level表示节点的深度，如果深度超过最大深度则跳出循环为4
                continue            
            if nt.score >= EPSILON2:  #置信度高，不用再分解
                if nt.level == 0:
                    FLAG = FLAG + 1
                    answer_final = self.force_answer(root.question, root.rationale)
                    return answer_final, root.rationale, FLAG
                   
                N.put(Node(level=nt.level + 1, question="", rationale="", score=0))  #不用再分解
                prev_nodes = self.extract(nt.question, nt.rationale, node_memory)
                print("prev_nodes:", [n.question for n in prev_nodes])
                for n_prev in prev_nodes:
                    if n_prev == root:
                        n_prev.rationale = self.update(nt.question, nt.rationale,
                                                n_prev.question, n_prev.rationale)   #更新前面节点的推理过程
                    if n_prev.level == 0:
                        n_prev.score = self.check_score(n_prev.question, n_prev.rationale)
                        if n_prev.score >= EPSILON2:
                            answer_final = self.force_answer(root.question, root.rationale)
                            return answer_final, root.rationale, FLAG
                    n_prev.question = self.Rewrite(nt.question,nt.rationale,n_prev.question)  #重写前面节点的问题
                    print("rewritten question:", n_prev.question)
                continue
            sub_questions = self.Decompose_new(nt.question)  #分解子问题 
            print(sub_questions)
            if len(sub_questions) > MAX_BRANCH:
                sub_questions = sub_questions[:MAX_BRANCH]  #选择前面的子问题
            new_nodes = []
            for sub_q in sub_questions:  #为每个子问题创建一个新的节点
                new_node = Node(level=nt.level + 1, question=sub_q)  #创建新节点，层数为当前层数的下一层，question为当前的父问题
                N.put(new_node)   #放进去新的节点
                new_nodes.append(new_node)   
                node_memory.append(new_node)  #将新的节点加入到节点记忆库中
        print("root.rationale",root.rationale)
        
        answer_final = self.force_answer(root.question, root.rationale)
        
        return answer_final,root.rationale, FLAG  #返回根节点的推理过程，作为最后的答案           
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
        # info_summary = "" 
        # round_count = 0
        # current_query = question
        # retrieval_history = []
        # last_contexts = []  
        # dependency_analysis_history = []  
        # retrieved_chunks_set = set() if self.filter_repeats else None  # Track retrieved chunks if filtering
        
        # print(f"\n\n{Fore.CYAN}{self.MODEL_NAME} answering: {question}{Style.RESET_ALL}\n\n")
        
        # #===============================================
        # #== Stage 1: warm up retrieval ==
        # if self.filter_repeats:
        #     new_contexts = self._retrieve_with_filter(question, retrieved_chunks_set)
        #     for chunk in new_contexts:
        #         retrieved_chunks_set.add(chunk)
        # else:
        #     new_contexts = self.retrieve(question)
        # last_contexts = new_contexts  
        # info_summary = self.refine_summary_with_context(
        #     question, 
        #     new_contexts, 
        #     info_summary
        # )

        # analysis = self.warm_up_analysis(question, info_summary)

        # if analysis["can_answer"]:
        #     # In this case, the question can be answered with simple fact retrieval, without any dependency analysis
        #     print(f"Warm-up analysis indicate the question can be answered with simple fact retrieval, without any dependency analysis.")
        #     answer = self.generate_answer(question, info_summary)
        #     # Reset dependency analysis history for simple questions
        #     self.last_dependency_analysis = []
        #     return answer, last_contexts, round_count
        # else:
        #     logger.info(f"Warm-up analysis indicate the requirement of deeper reasoning-enhanced RAG. Now perform analysis with logical dependency graph.")
        #     logger.info(f"Dependencies: {', '.join(analysis.get('dependencies', []))}")

        #     # sort the dependencies, by first constructing the dependency graphs, then use topological sort to get the sorted dependencies
        #     sorted_dependencies = self._sort_dependencies(analysis["dependencies"], question)
        #     dependency_analysis_history.append({"sorted_dependencies": sorted_dependencies})
        #     logger.info(f"Sorted dependencies: {sorted_dependencies}\n\n")
        # #===============================================
        # #== Stage 2: agentic iterative retrieval ==
        # idx = 0 # used to track the current dependency index

        # while round_count < self.max_rounds and idx < len(sorted_dependencies):
        #     round_count += 1
            
        #     current_query = sorted_dependencies[idx]
        #     if self.filter_repeats:
        #         new_contexts = self._retrieve_with_filter(current_query, retrieved_chunks_set)
        #         for chunk in new_contexts:
        #             retrieved_chunks_set.add(chunk)
        #     else:
        #         new_contexts = self.retrieve(current_query)
        #     last_contexts = new_contexts  # Save current contexts
            
            
        #     # Generate or refine information summary with new contexts
        #     info_summary = self.refine_summary_with_context(
        #         question, 
        #         new_contexts, 
        #         info_summary
        #     )
            
        #     logger.info(f"Agentic retrieval at round {round_count}")
        #     logger.info(f"current query: {current_query}")
            
        #     analysis = self.dependency_aware_rag(question, info_summary, sorted_dependencies, idx)

        #     retrieval_history.append({
        #         "round": round_count,
        #         "query": current_query,
        #         "contexts": new_contexts,
        #     }) 

        #     dependency_analysis_history.append({
        #         "round": round_count,
        #         "query": current_query,
        #         "analysis": analysis
        #     })

        #     if analysis["can_answer"]:
        #         # Generate and return final answer
        #         answer = self.generate_answer(question, info_summary)
        #         # Store dependency analysis history for evaluation access
        #         self.last_dependency_analysis = dependency_analysis_history
        #         # We return the last retrieved contexts for evaluation purposes
        #         return answer, last_contexts, round_count
        #     else:
        #         idx += 1
        
        # # If max rounds reached, generate best possible answer
        # logger.info(f"Reached maximum rounds ({self.max_rounds}). Generating final answer...")
        # answer = self.generate_answer(question, info_summary)
        # # Store dependency analysis history for evaluation access
        # self.last_dependency_analysis = dependency_analysis_history
        # return answer, last_contexts, round_count