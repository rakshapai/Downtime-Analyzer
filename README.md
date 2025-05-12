# Downtime-Analyzer Production Assistant Cookbook

### 1. Purpose and Audience

This cookbook is designed for Solution Architects, AI Engineers, and technical leaders looking to implement sophisticated generative AI solutions to solve real-world business problems, moving beyond toy examples to create impactful applications. It focuses on a multi-agent AI system designed to reduce production downtime in a manufacturing environment—a scenario where every minute of inefficiency translates to significant financial losses and material waste.
The core of the solution discussed is a conversational AI solution that empowers operators and managers to troubleshoot and resolve downtime issues using natural language. This platform leverages a multi-agent architecture, where specialized AI agents interact not only with Large Language Models (LLMs) like OpenAI's GPT-4o via the Assistants API but also with other critical enterprise systems and AI algorithms. These include historical data logs, real-time sensor data, and optimization engines such as CPLEX.

**Why this matters:** Traditional factory dashboards and reports can overwhelm users with data but still require manual analysis which takes time. By asking a natural language question, a plant manager or engineer can get immediate answers — whether it’s querying last week’s stoppages, finding how to fix a machine, or simulating improvements. This system integrates with the company’s data sources and workflows (e.g. pulling data from SQL databases, retrieving information from equipment manuals, interfacing with maintenance ticket systems) to ensure answers are relevant and actionable in the real operational environment. The emphasis is on creating robust, scalable, and reliable solutions that deliver measurable business value, as exemplified by the described system which achieved over $10 million in annual savings for a major food manufacturer.
The audience will gain insights into:
*	Designing and implementing multi-agent AI systems (RAG, Code Generation, Report Generation)
*	Integrating LLMs with diverse data sources and external tools like SQL databases and optimization solvers.
*	Addressing challenges such as noisy data, user context management, and ensuring operational safety compliance.
*	Evaluating the performance and impact of such AI solutions.
*	Optimizing costs and preparing for production deployment.
  
**Audience:** This document is for technical leads and architects who will design and implement such an AI assistant. It assumes familiarity with concepts like LLM APIs, vector databases, and integration of APIs/DBs, but it explains the design choices and pitfalls with the help of use cases. By the end, a reader should understand how to build a multi-agent LLM application that works within an enterprise architecture — including how it routes user questions to different specialized agents, how those agents work under the hood, and how to evaluate and deploy the system for production use. 

#### 2. Use Cases Overview
In our downtime reduction assistant, we have multiple specialized LLM-powered agents, each designed to handle a category of user queries. An Orchestrator Service acts as a traffic controller: it analyzes each user question (through intent classification or rules) and routes it to the appropriate agent. The core use cases and corresponding agents include:

*	Data Querying (SQL Agent) – Handles analytical questions like “What were the top 10 stoppages last week?” by querying databases and summarizing results.
*	Troubleshooting Q&A (RAG Agent) – Handles maintenance and how-to questions like “How do I fix a jam on Conveyor 3?” by retrieving relevant info from manuals (Retrieval-Augmented Generation with citations).
*	Optimization & Simulation (CPLEX Agent) – Handles "what if" and optimization questions like “Simulate the yield change if we move product X to a different line” by formulating and solving an optimization or performing a simulation, possibly using an optimization engine (e.g. CPLEX).
  
Each agent uses the LLM in a different way (from writing SQL, to fetching documents, to generating code), but they all integrate with existing factory systems. The diagram below shows the high-level architecture and data flow of the solution, with the orchestrator, agents, data layers, and response formatter working together for each query.

![image](https://github.com/user-attachments/assets/c1ffce7d-fdb1-4875-8232-e66d113c9137)

Figure: High-level architecture of the multi-agent system. 
*The Orchestrator Service receives user questions (Step 2) from the web UI and routes them (Step 4) to the appropriate agent based on intent. 
The agent (SQL, RAG, CPLEX, etc.) may fetch data from relevant sources (Step 5: e.g., SQL DB, vector store, sensors) and returns an intermediate result (Step 6). 
The Response Layer (Final Answer, Guardrails, Recommendation services) parses and formats this result (Step 7) into a user-friendly answer (Step 8), potentially with follow-up suggestions. The full answer is sent back to the user (Steps 10–11). The system also stores conversation context in a memory store (Steps 3 and 9) so that follow-up questions can be understood in context.*

Below, we go through each use case in detail, including how the agent works (tech stack), an example conversation flow, implementation snippets, and key lessons learned.

#### Use Case 1 – SQL Agent – Querying Operational Data
##### 1A. Core Technology Stack
For analytical queries, the SQL Agent is responsible for converting natural language questions into database queries and then turning the query results into an insightful answer. The stack for this agent includes:
*	LLM (OpenAI GPT-4) for understanding the user question and generating SQL code as well as summarizing results. We use a few-shot prompting approach where the model is given examples of questions and corresponding SQL queries to guide its output.
*	Database Connector/Wrapper to safely execute the generated SQL on the company’s production database (which contains records of stoppages, downtime, etc.). The database is typically a SQL database (e.g., PostgreSQL or SQL Server) that stores structured data like factory name, line number, shift, machine, downtime minute etc.
*	Integration with maintenance system (ServiceNow API): In our scenario, each downtime entry might have associated maintenance ticket details. The SQL Agent can join or enrich data from the downtime database with data from ServiceNow (through an API call) to pull in extra details like ticket IDs, root cause analysis, or resolution status. This happens in the background after the SQL query, before formulating the answer.
*	Conversation memory via a Redis cache, to retain context across turns. For example, if the user’s follow-up question omits the factory name (assuming the context from the previous query), the orchestrator uses memory to inject “Lincoln Factory” as context for the next query.
  
All these components are orchestrated such that the user’s question is translated to a DB query, executed, and then the results are fed back into the LLM to produce a friendly answer. The Final Answer Service then formats any tabular data into Markdown tables and the Recommendation Service may add follow-up questions.
 
##### 1B. Conversation Flow
Example: The user asks Q1: “Can you show me the top 10 stoppages at Lincoln Factory last week?”. Here’s how the conversation flows through the system. 

![image](https://github.com/user-attachments/assets/64cf9777-fb1c-4520-bffa-d3f8b08ae502)

1.	**Orchestrator (Intent Routing):** The question is classified as a data query (intent: Analytics.Query). The Orchestrator routes it to the SQL Agent.
2.	**SQL Agent – Query Generation:** The agent’s prompt includes a brief instruction and possibly examples, such as: “You are a data analyst. Given a question, output an SQL query for our downtime database. Example: Q: What were the total downtime minutes for Factory A last month? A: SELECT Factory, SUM(Downtime) ...”. The user’s query is appended, and the LLM produces an SQL statement. The agent might also generate multiple queries if needed (e.g., if “last week” requires computing date range, or separate queries to gather additional info).
Note: Hierarchy handling logic to parse entities like factory, line, shift, and time frame from the question. For example, if the user asks “... at Lincoln Factory last week”, the agent (or a preprocessing step) interprets that as factory = 'Lincoln' and a date filter for last week. This ensures the generated SQL uses the correct filters. The orchestrator can supply known entity values (like standardized factory names or date ranges) to the prompt.
3.	**Database Execution:** The orchestrator (or a function call mechanism) executes this SQL against the SQL DB if the results are not present in the cache. The query result is a list of the top 10 stoppage records for Lincoln factory in the last week. Each record has details like which line and machine, how many minutes of downtime, product affected, and a short comment describing the issue.
4.	**Result Parsing and Answer Generation:** The SQL Agent now has the raw data and enriched details. Another LLM prompt is used to compose the answer which also contains the business context while generating the response. Answers can also be formatted as table, chart or just plain text depending on the user question type. 
5.	**Recommended Follow-ups:** The Recommendation Service appends a few follow-up questions relevant to this data. There is a separate service that keeps track of the current user question, relevant entities, hierarchical logic and an LLM call to generate further exploration question. 
6.	**Follow-up (Q2):** The user picks a recommended question: “What are some issues causing the downtime?” Since this is a follow-up in the same conversation, the orchestrator knows the context is still the Lincoln Factory’s last week data. It routes to the SQL Agent again, but this question is more analytical (it asks for issues causing downtime, likely summarizing the patterns in the data already retrieved). The SQL Agent might not even need a new DB query if it reuses the results from Q1 stored in memory. Instead, it can directly analyze the list of stoppages from Q1. The LLM is prompted to summarize the key issues.
7.	**Maintaining Context:** Notably, Q2 did not explicitly mention “Lincoln” or time range, but the system assumed the user was still referring to the last query’s context. The conversation memory (in Redis) provided the needed context (factory=Lincoln, week=last week, top 10 events list), so the agent focused on those. If the user had instead asked a completely new question unrelated to Q1, the orchestrator would treat it independently.

This flow demonstrates how the SQL Agent goes from a user question to data and to a natural explanation, integrating multiple systems seamlessly. The user essentially gets a mini-report in seconds.

##### 1C. Implementation (Code Snippets)
Below are illustrative code snippets for implementing the SQL Agent logic using the OpenAI API and Python. We assume we have a function execute_sql(query) that runs a SQL query and returns results as a list of rows or a panda DataFrame, and a function get_issue_details(event) that fetches additional maintenance details for a downtime event.
First, we prepare the LLM prompt for query generation with a few-shot example to guide the model:

``` import openai

# Few-shot examples for prompt
examples = [
    {"role": "user", "content": "Total downtime at Factory A last month?"},
    {"role": "assistant", "content": "SQL: SELECT Factory, SUM(Downtime) AS TotalDowntime " "FROM DowntimeEvents WHERE Factory = 'Factory A' "
                                     "AND EventDate >= DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month') "
                                     "AND EventDate < DATE_TRUNC('month', CURRENT_DATE) "
                                     "GROUP BY Factory;"},
    {"role": "user", "content": "Top 5 issues on Line 2 yesterday?"},
    {"role": "assistant", "content": "SQL: SELECT IssueType, COUNT(*) as Count, SUM(Downtime) as TotalMins "
                                     "FROM DowntimeEvents WHERE Line = 2 AND EventDate = CURRENT_DATE - 1 "
                                     "GROUP BY IssueType ORDER BY TotalMins DESC LIMIT 5;"}
]

def generate_sql_query(natural_language_query: str) -> str:
    system_msg = {"role": "system", "content": "You are a helpful assistant that translates user questions into SQL queries. Only provide the SQL code prefixed by 'SQL:'."}
    user_msg = {"role": "user", "content": natural_language_query}
    # Combine system message, examples, and the new query
    messages = [system_msg] + examples + [user_msg]
    response = openai.ChatCompletion.create(
        model="gpt-4", 
        messages=messages,
        temperature=0.2  # low temperature for deterministic output
    )
    sql_code = response['choices'][0]['message']['content"]
    # Remove the "SQL:" prefix for execution
    return sql_code.replace("SQL:", "").strip()
```
In this snippet, we instruct the model to act as a translator to SQL. We include example pairs so it learns the style. For a query like “Show me the top 10 stoppages at Lincoln Factory last week,” the model should produce an appropriate SQL SELECT statement with filters for factory and date.

Next, execute the generated query and integrate additional data:

``` user_query = "Can you show me the top 10 stoppages at Lincoln Factory last week?"
sql = generate_sql_query(user_query)
results = execute_sql(sql)  # returns e.g. a pandas DataFrame

# Enrich results with maintenance details
detailed_results = []
for row in results.itertuples():
    issue_info = get_issue_details(row)  # e.g., fetch maintenance ticket or description
    detailed_results.append({
        "Factory": row.Factory,
        "Line": row.Line,
        "Shift": row.Shift,
        "Machine": row.MachineName,
        "Downtime_mins": row.Downtime,
        "Product": row.Product,
        "Issue": row.Comments,        # short issue code
        "Details": issue_info         # longer description from maintenance system
    })
```
Now we have ``` detailed_results ```, a list of dictionaries including both the original data and enriched details. We feed this into the LLM again to generate a nicely formatted answer with a table and explanations:

``` # Convert detailed_results to a Markdown table and explanatory text via LLM
table_header = "| Factory | Line | Shift | Machine | Downtime (mins) | Product | Issue |\n| --- | --- | --- | --- | --- | --- | --- |\n"
table_rows = ""
for item in detailed_results:
    table_rows += f"| {item['Factory']} | {item['Line']} | {item['Shift']} | {item['Machine']} | {item['Downtime_mins']} | {item['Product']} | {item['Issue']} |\n"

# Prepare a prompt for formatting the final answer
format_system_msg = {"role": "system", "content": "You are a data analyst assistant that formats query results into an explanation with a table. Focus on business insights such as trends in downtime, impact on revenue or patterns around downtime for a specific product/line/shift "}
user_prompt = (
    "The user asked: 'Can you show me the top 10 stoppages at Lincoln Factory last week?'.\n"
    "We have the results and details. Present the answer as a summary, followed by a table of results and detailed explanations for each issue:\n\n"
    + table_header + table_rows +
    "\nProvide a brief summary of total downtime and highlights, then list each issue with 'Details: ...'."
)
format_user_msg = {"role": "user", "content": user_prompt}

response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[format_system_msg, format_user_msg],
    temperature=0.3
)
final_answer = response['choices'][0]['message']['content']
print(final_answer)  # This would be sent to the user
```
In practice, you might combine some of these steps or use a single prompt that both generates the query and answers the question (using OpenAI’s function calling feature, for instance, to let the model request data). But the above breakdown makes it easier to monitor each part: generating SQL and formatting output. The key implementation points are:
*	Use a deterministic LLM call (low temperature) for SQL generation to reduce errors.
*	Post-process the SQL to avoid any harmful commands (ensuring it’s read-only, no drops or modifications).
*	Use the LLM to turn data into narrative, which is where the real value-add comes in for the user. Make sure you provide the prompt with enough business/use case context so that it can summarize with the appropriate level of detail. 
*	If you run into issues with LLM token limits which is often the case, you can save the prompt Q&A examples in a vector DB and use similarity match to fetch most relevant k samples from the DB based on the user question.  

##### 1D. Takeaways (Lessons Learned)
Implementing the SQL Agent taught us several important lessons:
*	**Provide schema and examples to the LLM:** We supplied table, column names, table relationships as well as syntax notes (either in the system prompt or few-shot examples) so it generates valid queries. It’s wise to include a brief schema description in the prompt context (e.g., “Table DowntimeEvents columns: Factory, Line, Shift, MachineName, Downtime, Product, Comments, EventDate”).
*	**Guard against SQL injection or errors:** Even though the LLM writes the SQL, treat it as untrusted input. We validate the generated SQL — for example, restrict to a whitelist of SELECT statements or use prepared statements. The execution function should have safe-guards (read-only DB user, query timeout limits) to prevent any accidental harm. Also handle exceptions (if the SQL is malformed or times out, catch it and have the assistant respond gracefully) to provide an error message or a default response back to the user. 
*	**Hierarchy and context handling:** Users might not specify all filters every time. The system should infer or follow up. In our case, if the user omitted “Lincoln Factory” but previously asked about it, we assumed that context. This makes the conversation natural. However, be cautious: if context is stale or user changes topic, don’t misapply the old context. Our Orchestrator uses simple rules – if the new question is related (e.g., a follow-up) use last context, otherwise reset context or ask for clarification. Use finer details like user conversation topics and entities that can be passed to the below sub-routines to maintain quality of output. 
*	**Combine data with domain knowledge:** The enriched explanations (from maintenance logs) made the answer far more useful than raw numbers. This shows the benefit of integrating multiple data sources. It’s important to maintain consistent identifiers between systems (e.g., machine names or event IDs) so that an agent can join data from the SQL DB with the maintenance system seamlessly.
*	**Don’t assume one-shot perfection:** The SQL Agent sometimes might not get the query perfect on the first try (especially for very complex queries). We included some simple post-checks. For instance, if the model’s SQL missed a date filter for “last week,” the Orchestrator could detect it (by noticing no time clause) and append a default date range or ask the model to refine the query. These guardrails ensure the answer is correct and relevant.

#### Use Case 2 – RAG Agent – Troubleshooting with Manuals Q&A

##### 2A. Core Technology Stack
The RAG (Retrieval-Augmented Generation) Agent provides step-by-step troubleshooting guidance by grounding its answers in your own documentation (equipment manuals, SOPs, service reports). In production, we implement this with the OpenAI Assistants API’s file search tool, which handles chunking, embedding, and metadata filtering for you.
1.	**Automatic File Ingestion & Chunking**
*	Upload manuals (PDFs, DOCXs, etc.) via the Assistants API with purpose="assistants".
*	The service automatically splits each file into ∼800-token chunks (with overlaps) and generates embeddings (using text-embedding-3-large).
*	Chunks inherit file-level metadata (e.g. machine: "conveyor_3", category: "Food Products"), which you provide at upload or annotate afterward.
2.	**Hybrid Retrieval Logic**
  *	Metadata filtering: Before semantic search, narrow scope to only those chunks whose metadata matches extracted entities. For “How to fix the jam in Conveyor 3?”, filter to machine="conveyor_3".
  *	Semantic similarity: The API then ranks the filtered chunks by embedding similarity to your query, returning the top K (e.g. K=5–10). This combination of filters + embeddings ensures both precision and recall without rolling your own vector store.
3.	**LLM Synthesis (GPT-4 or GPT-4o)**
*	Pass the retrieved passages directly into a completion call (or let the assistant invoke file_search internally).
*	Use a prompt like:
“The user asked: ‘How do I clear a jam on Conveyor 3?’
Here are relevant excerpts from the Conveyor 3 maintenance manual:
```[
Excerpt 1: …]
[Excerpt 2: …]  
```
Please provide a safe, step-by-step procedure. Cite the manual where appropriate. If you need more information, ask the user.”
*	The model generates an answer grounded in the excerpts and includes citations (e.g. “(see Manual 5.2)”).
4.	**Citation & Safety Guardrails**
*	Answers include inline citations or links like “According to Conveyor 3 Manual (pp. 45–46)…” so operators can verify the source.
*	Safety-critical passages (lock-out/tag-out, PPE) are prioritized by tagging those sections with ```safety_level: "critical"``` metadata and bumping them to the top in retrieval.
This stack ensures your RAG Agent never hallucinates procedures—it uses your own, up-to-date manuals, scoped by metadata filters, and synthesized by a best-in-class LLM.

##### 2B. Conversation Flow
**User:** “How do I fix the jam in Conveyor 3?”

**1.	Orchestrator** classifies this as a **Troubleshooting** query and extracts entities. 
When a user question arrives, the orchestrator agent first determines if it requires internal documentation. For instance, a query about a specific error code or machine symptom would be routed to the RAG retrieval process, whereas a general question might be handled by a different agent or a web search tool. This logic can be implemented via prompt-based classification or rules (e.g. keywords that map to internal knowledge domains).
```{ "machine": "conveyor_3", "issue": "jam" }```
**2.	Assistant File Search call** - If internal knowledge is needed, the orchestrator triggers the RAG agent (or instructs the assistant directly) to perform a file-based search. Using the Assistants API’s tool calling, the agent queries the vector store for relevant content. This can be done in two ways: (1) by calling the vector store search API directly to fetch documents, or (2) by invoking the file_search tool as part of a completion. The second approach lets the model handle retrieval within a single API call. In either case, the agent can apply metadata filters for targeted search (e.g. restrict to machine = "conveyor_3" documents). The result of this retrieval step is a set of relevant text chunks related to the query.

**RAG Agent** calls the Assistants API’s **file_search** tool, specifying:
```
{
  "tool": "file_search",
  "vector_store_ids": ["manufacturing_docs"],
  "filters": { "type": "eq", "property": "machine", "value": "conveyor_3" },
  "query": "clear jam on conveyor 3"
}
```
3.**Disambiguation with Tools:** If the query is ambiguous or lacking key details, the assistant can leverage its tool-calling capabilities to disambiguate. For example, if a user asks, “The motor is overheating, what should I do?” without specifying which machine, the agent might first perform a broad file search (across all machine manuals) and find multiple references. Detecting this, it could call a custom tool or use a follow-up prompt to ask the user for clarification (“Which machine or line are you referring to?”). The Assistants API supports multi-step tool interactions, so the agent could also automatically refine the search (e.g. searching for the error in each machine’s context) and decide which result is most relevant. This ensures the RAG agent zeroes in on the correct context before drafting an answer.
**4.Answer Synthesis:** Finally, the assistant uses the retrieved information to compose a helpful answer. The relevant document excerpts are injected into the model’s context (either implicitly by the file_search tool or explicitly by your code), and the model’s response is thereby grounded in those facts. The assistant will combine the retrieved knowledge with its own reasoning and language abilities to generate a concise, context-aware answer. In a production setup, you might also have the assistant cite the source of the information or quote the documentation. The orchestrator then returns this answer to the user. Importantly, all of this happens behind the scenes in a seamless flow, so the end-user only sees a coherent answer, not the retrieval steps.
*	The API returns the top 5 most relevant manual chunks, each with an internal score and metadata.
*	The agent prompts GPT-4 with those excerpts plus instructions to:
  * Emphasize safety steps
	* List a clear procedure
  * Cite the manual
*	Assistant replies:
  * **Safety First:** Ensure the conveyor is stopped and power isolated (lock-out/tag-out).
  * Procedure:
  * •	Press the emergency-stop button on Conveyor 3.
  * •	Isolate power from the main switch (see Manual §3.1).
  * …
  * Source: Conveyor 3 Maintenance Manual (pp. 45–46)

##### 2C. Implementation Details
```
from openai import OpenAI

client = OpenAI()

# 1. (One-time) Upload each manual with metadata
file_resp = client.files.create(
    file=open("Conveyor3_Manual.pdf", "rb"),
    purpose="assistants",
    metadata={"machine": "conveyor_3", "category": "Food Products", "safety_level": "high"}
)
client.vector_stores.files.create_and_poll(
    vector_store_id="manufacturing_docs",
    file_id=file_resp.id
)

# 2. Handle a troubleshooting query
query = "How do I fix the jam in Conveyor 3?"
response = client.responses.create(
    model="gpt-4o",
    input=query,
    tools=[{
        "type": "file_search",
        "vector_store_ids": ["manufacturing_docs"],
        "filters": {
            "type": "eq",
            "property": "machine",
            "value": "conveyor_3"
        }
    }]
)

# 3. Extract and display the assistant’s answer
answer = response.output[-1].content[0].text
print(answer)
```
**Key Points:**
*	**No manual chunking required:** the API handles it for you.
*	**Metadata filters** guarantee you only search the relevant machine’s documents.
*	**Hybrid search (keyword + embeddings**) maximizes both coverage and precision.
*	**Safety-tagged content** is surfaced first to ensure compliance.

With this setup, your RAG Agent is both powerful and easy to maintain—updates to manuals or metadata are as simple as uploading new files, and the Assistants API takes care of the rest.

##### 2D. Takeaways and Lessons Learned

*	**Document Preprocessing:** To maximize retrieval quality, ensure that documents are prepared for ingestion. Convert scanned PDFs or images of text into machine-readable text via OCR, since the file search tool has limited ability to parse images or non-text content in documents. It’s also wise to break up very large documents by section or topic if possible (the automatic chunking will handle splitting, but organizing content logically can aid relevance). Remove any content that is outdated or not meant to be used, so it doesn’t accidentally influence answers. Essentially, treat your knowledge base as a curated library: up-to-date, relevant, and in a text format that the AI can read. 
*	**Metadata Usage:** Take advantage of metadata to label and organize your knowledge base. Decide on a schema for your metadata (e.g. machine, category, version, safety_level) and apply these tags during or after file upload. Currently, you may attach metadata by providing it when adding files to the vector store or by naming files in a consistent pattern that your system can interpret. Using metadata filters in queries will then allow the RAG agent to, for example, only retrieve chunks from Machine A’s manuals when the question is about Machine A. This selective retrieval greatly improves response precision. The Responses API’s retrieval tooling supports such filtering to narrow down search scopecookbook.openai.com. (Note: Ensure your metadata values are consistent and error-free; a typo in a tag could exclude relevant data from a search.)
*	**Safety Tagging:** Integrate safety considerations into your retrieval pipeline. In a manufacturing context, some documents might contain sensitive information or potentially hazardous instructions. You can tag these documents (or specific sections) with a safety classification, and then program the assistant to handle them carefully.
For example, a tag like safety: "restricted" could signal the agent to either avoid using that content or to include warnings in its answer. OpenAI’s assistant platform includes built-in refusal conditions and content moderation checks to prevent disallowed content from being output. Leverage these by ensuring your documents are in line with policy (no instructions that violate usage guidelines), and by perhaps adding an extra layer of filtering – for instance, run retrieved text through a moderation model before using it. 
*	**Continuous Updates:** Keep the vector store updated as your knowledge evolves. The Assistants API allows you to append new files (which get chunked and indexed on upload) and remove or replace outdated ones. In a production deployment, you might set up a regular job to ingest the latest manuals, incident reports, or knowledge base articles. The underlying vector index can handle continuous growth, and you can also rebuild it if major changes occur. OpenAI provides up to 1GB of free vector storage (with a usage-based fee beyond that), which is plenty for a large document.

#### Use Case 3 – CPLEX Agent – Optimization and “What-If” Simulations

##### 3A. Core Technology Stack
The **CPLEX Agent** (named after IBM CPLEX, a solver for optimization problems) is designed for scenario analysis and optimization queries. These are questions where the user is asking for a **decision or prediction** rather than a straightforward fact. For example: “What if we move product A to a different line, how would it affect output?” or “Optimize the maintenance schedule to minimize downtime.” Solving these often requires formulating a mathematical model or running a simulation. The stack for this agent is a bit special:

*	**LLM (GPT-4)** for interpreting the question and generating a structured solution approach. We fine-tuned or few-shot trained the model to identify key components: assumptions, constraints, and objectives from the user query. The model was also trained on the mathematical formulations necessary to the problem space. Essentially, the LLM acts as a translator from natural language into a pseudo-code or actual code for solving the problem.
*	**Optimization/Solver Engine:** Once the problem is formulated, we use a solver to compute the results. This could be the CPLEX engine (if solving an ILP or LP), or a simpler Python calculation if the scenario is not too complex. In many cases, the agent generates Python code using libraries like PuLP (an open-source LP solver) or even just arithmetic for simple scenarios. The code is then executed within a sandboxed environment.
*	**Domain model or data access:** The agent might need data to run the simulation. For the example “simulate yield change if Maggi moves to Conveyor 2”, it needed the downtime of Conveyor 3 and Conveyor 2, and the production rate for Maggi. It can obtain these from previous conversation context (Q1 had the downtime info) or query the database directly for those parameters. We allow the agent to call the SQL Agent or use a cached result for such data. This is an example of agents collaborating: the CPLEX agent might internally invoke a data fetch tool.
*	**Trained scenario templates:** We prepared several example problems and solutions for the LLM to learn how to respond. For instance, a few-shot prompt might include: “User asks: ‘If machine X runs at Y% capacity, what is the output?’ -> Assistant: (assumptions: baseline capacity=..., constraint: output cannot exceed..., calculation: ... result: ...).” This teaches the model to structure its answer as: list assumptions, show calculations or logic, then give the outcome.
*	**Interactive decision integration:** Often these optimization answers lead to an action. We integrated with the workflow to allow users to take next steps. In our case, after computing the yield impact, the assistant asks “Would you like to submit this proposed change for review and approval?” If the user clicks Yes, the system would create a recommendation record (via an ERP or maintenance planning system, MES) for a supervisor to review the proposal of moving Maggi to another conveyor. This closes the loop from analysis to action.
  
Overall, the CPLEX agent is like having a data scientist or a scientist in the loop of the conversation – it can run numbers and optimization logic on the fly and present results which now takes minutes instead of days. 

##### 3B. Conversation Flow
Picking up the conversation, the user now asks Q4: **“Simulate the yield change if Maggi is replaced to Conveyor 2 instead of 3.”** This is a hypothetical scenario question. Here’s the flow:

![image](https://github.com/user-attachments/assets/9fba6977-427d-43c3-a6dc-79d7194addb0)

**1.	Orchestrator (Intent Classification):** The question is identified as a Simulation/Optimization intent. The presence of words like “simulate” or “what if” and domain terms (Maggi, conveyors) triggers the CPLEX Agent.
* **2.	Understanding the Problem:** The CPLEX Agent first parses what the user is really asking. “Maggi is replaced to Conveyor 2 instead of 3” means currently Maggi is on Conveyor 3 (where we saw an issue: 20 min downtime due to belt slippage from Q1 data). The user wants to know what happens if Maggi ran on Conveyor 2. Implicitly, Conveyor 2 was running Coffee-Mate last week and had 5 min downtime (from Q1 data). So the problem is: compare Maggi on Conv3 vs Conv2 in terms of downtime and yield. The agent identifies:
  *	**Assumptions:** e.g., “The belt slippage issue is specific to Conv3 and would be avoided by moving Maggi. Conveyor 2 can handle Maggi similarly to Coffee-Mate’s operations. We’re looking at the same time period (last week).”
  *	**Constraints:** not a strict optimization problem here, but constraints/considerations might be “Conveyor 2 had a minor sensor failure (5 min) which could also affect Maggi if moved. Production rate for Maggi is 100 units/10min on either conveyor.”
  * **Objective:** simulate yield change (so basically compute net gain or loss in production).
* **3.	Data Fetching:** To perform the simulation, the agent pulls needed numbers:
  *	Downtime on Conv3 for Maggi (last week) = 20 minutes (from earlier context).
  *	Downtime on Conv2 (same period) = 5 minutes (Conv2 was running Coffee-Mate, but we assume similar downtime if Maggi were there).
  *	Production rate of Maggi on a conveyor = e.g. 100 units per 10 minutes (or 600 units/hour) – this might be a known standard or could be fetched from an ERP system if different products have different speeds. In our conversation, they assumed 100 units/10min. If the context wasn’t in memory, the agent could query the database: e.g., SELECT Downtime FROM DowntimeEvents WHERE Product='Maggi' AND Conveyor='3' ... etc. But since Q1 provided these, it likely reused them.
* **4.	Computation:** The agent now performs the simulation: If Maggi were on Conveyor 2, presumably it avoids the 20 min belt issue, but would still have been affected by the 5 min sensor issue that happened on Conv2. So net downtime saving = 15 minutes. With a production rate of 600 units/hour, 15 minutes less downtime = 150 more units produced.
The agent decides to present this in a narrative form. It generates an answer that includes:
  *	A recap of key data: “Conveyor 3 (Maggi) had 20 min downtime (belt slippage), Conveyor 2 (Coffee-Mate) had 5 min downtime (sensor failure).”
  *	A list of **Simulation Assumptions** clearly enumerated (this was shown in the answer: belt issue avoided, sensor issue remains, etc.).
  *	The **Potential Outcome:** stating that moving Maggi could save ~15 min downtime, translating to ~150 extra units of Maggi in that period.
  *	A caveat that this is a simplified analysis and a full study would consider more factors (the assistant explicitly said this to manage expectations).
* **5.	Technical Details (Optional):** The assistant also provided a “Show Technical Details” section with the raw calculation and even a Python code snippet that was created and executed in the sandbox environment to run the scenario. 
* **6.	Follow-up Action:** After giving the result, the assistant asks “Would you like to submit this proposed change (Maggi from Conv3 to Conv2) for further review and approval by a supervisor?” If the user said Yes, the system could trigger the Report/Action Agent – for example, log this recommendation to a database or create a ticket for engineering to evaluate re-routing Maggi production through external integration. You can either store the ticket details in the database or call the ServiceNow integration live -> all these calls can be done in parallel since they are asynch calls and don’t impact the conversation flow. 
If No, then it’s just hypothetical and no action taken.
* **7.	Further Recommendations:** The conversation doesn’t end here; the assistant suggests other forward-looking questions: “Analyze impact on other product lines” or “Compare this change to other potential improvements” or “Generate a report for historical trends.” These prompt the user to consider broader analysis or to request a formal report. The Report Agent (another agent not detailed in this cookbook) could, for instance, generate a PDF report of downtime trends.

In this flow, the user essentially had a data scientist quickly simulate a scenario and then got the option to turn that into action. This showcases how generative agents can not only inform but also assist in decision-making processes.

##### 3B. Implementation (Code Snippets)
Implementing the CPLEX agent involves prompting the LLM to produce either a direct answer with calculations or actual code. In development, we tried both approaches:

*	**Direct Calculation:** For simpler “what-if” questions, the LLM can just do the math in its head (it’s surprisingly good at basic arithmetic, especially if the numbers are simple).
*	**Code Generation:** For complex optimization (e.g., scheduling, or if we wanted to truly use CPLEX for an ILP), we have the LLM generate Python code that we then execute.

For this example, we’ll illustrate a hybrid: ask the LLM to produce the calculation steps and code, then execute the code to double-check the result.

```
import openai
# 1. User question and data (from memory or DB)
user_question = "Simulate the yield change if Maggi is replaced to Conveyor 2 instead of 3."
downtime_C3 = 20   # minutes of downtime for Maggi on Conveyor 3 last week
downtime_C2 = 5    # minutes of downtime for Conveyor 2 in same period
production_rate = 600  # units per hour for Maggi

# 2. Construct a rich, detailed prompt
detailed_prompt = """
You are an expert operations researcher and optimization assistant. A plant manager has asked you to simulate the yield change 
when the Maggi product is re‐routed from Conveyor 3 to Conveyor 2 instead of its current assignment. 
Please follow these instructions exactly:
1. **Restate the Problem**: Begin by summarizing the scenario in your own words, including the downtime values and production rate.  
2. **List Your Assumptions**: Clearly enumerate at least three assumptions you are making (e.g., production rate fixed, downtime on new conveyor equivalent to historical rates, switching time negligible).  
3. **Define the Objective**: State the optimization objective (for example, “maximize total units produced given downtime”).  
4. **Specify Constraints**: List all constraints in mathematical form (e.g., `units_produced <= production_rate * (60 - downtime)` for each conveyor).  
5. **Formulate LP Model**: Write production variables, objective function, and constraints as a linear program using the Python `docplex` library.  
6. **Solve & Report**: Solve the model, then report:  
   - Net downtime difference (minutes)  
   - Units produced before and after re‐routing (per hour)  
   - Yield change (units per hour)  
7. **Output Format**:  
   - A short “Solution Summary” paragraph with the key numeric results.  
   - A Python code block using `docplex.mp.model.Model` that builds and solves the LP.  
   - Clear comments in code explaining each step.
User Question: """ + user_question + f"""
- Downtime on Conveyor 3: {downtime_C3} minutes  
- Downtime on Conveyor 2: {downtime_C2} minutes  
- Production rate for Maggi: {production_rate} units/hour  

# 3. Call the LLM
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": "You are a senior operations research engineer. Write crisp, correct docplex code and explanations."},
        {"role": "user",   "content": detailed_prompt}
    ],
    temperature=0.0
)
# 4. Extract and print
answer = response.choices[0].message.content
print(answer)
```
After execution, we’d compare the result of the code with what the model stated. This is a form of validation to catch any arithmetic mistakes (though in our testing, GPT-4 was accurate with these simple numbers).

If the question were a more complex optimization like “Optimize the maintenance schedule for next week to minimize downtime while ensuring each machine gets serviced”, the approach would differ:

  **	We’d have the GPT-o3 LLM formulate a linear optimization problem (define decision variables for maintenance timing, objective minimize downtime, constraints like each machine maintenance = 1 time slot, etc.).
  **		The LLM could output a Python code using docplex library to define and solve this ILP.
  **		Our system would run that code to get the solution and then translate it into an answer (e.g., which day to schedule each machine’s maintenance). While powerful, this approach requires careful validation (the code could have bugs or not solve if model missed a detail).

In summary, the CPLEX agent implementation combines LLM reasoning with actual computational tools to ensure the answers are not just plausible, but computed. We encapsulated this in a service that can run the LLM’s generated code in a sandbox, catch errors (e.g., if the model code throws an exception), and return the results safely.

##### 3C. Takeaways (Lessons Learned)
From developing the CPLEX agent, we gathered several valuable insights:

*	**NLP to Math translation is challenging:** Getting the model to correctly extract constraints and objectives from a question is hard. We found that providing explicit examples in the prompt of “how to list assumptions, constraints, etc.” improved reliability. It’s important to have the model restate the problem in its own structured way (as it did with the assumptions list) – this way, the user also sees transparently what is assumed. This manages expectations and allows the user to correct any wrong assumptions. We also provided the model with enough data context such that which columns would be used as constraints, which can be used in the objective function. This ensured that the downstream code that gets generated is accurate. 
*	**Keep it simple when possible:** Not every scenario needs a heavy ILP solve. In fact, many “what-if” questions can be answered with simpler math or heuristics. We taught the agent to recognize when a straightforward calculation would do. This avoids overkill and reduces runtime. Save the true CPLEX power for when the question explicitly demands optimization across many variables. You can switch the LLM here and use a light-weight one like 4.1 for simpler use cases. 
* **Validation of LLM-generated code:** Treat code from the LLM with skepticism, just like SQL. We wrote a validator that checks for risky operations in the code (no file writes, no external calls, etc., since we only expect math/solver code). We also limited execution time – if a generated optimization tries something too complex, we stop it to prevent hanging. In one test, the model wrote a brute-force search that took too long; the system had to kill it and respond with an error message. We then refined the prompt to prefer efficient methods or give up if the problem is too large.
* **Floating point and units:** The model might not automatically handle unit conversions or rounding as a human would. In the yield simulation, it assumed an hour has 60 minutes, which is fine, but in other contexts (e.g., “percent improvement”), we saw it sometimes express answers in a confusing way (like 0.15 vs 15%). We added post-processing to format the results nicely (e.g., as percentages or whole numbers as appropriate) and to ensure consistency (units of measure clearly stated).
*	**User engagement with analysis:** We discovered users loved the transparency of showing the “Technical Details” (like code or formulas), but not everyone wants that in the main answer. Our UI design was to hide it behind a toggle. The content was produced by the LLM as part of the answer, annotated in a way that the front-end knows it’s extra detail. This approach could be generalized: the agent can produce a summary and a detailed analysis separately (using tools like Markdown headings or delimiters to split them).
*	**When LLM meets domain models:** We did consider using a trained ML model for predictions (for example, a regression model for yield vs downtime). If such a model existed, the agent could call it instead of doing a calculation. This would be like a “tool” the agent has. In our implementation, we primarily used analytical formulas. But in future, incorporating predictive ML models for things like “if we reduce downtime by 10%, what’s the impact on annual output?” could be integrated. The architecture allowed the agent to call out to such models via an API or function call if needed.
* **Collaboration between agents:** The CPLEX agent often required data (like downtime numbers) that the SQL agent could provide. Rather than duplicate that logic, we allowed a form of agent collaboration. In our design, either the orchestrator pre-fetched some data knowing the question context, or the CPLEX agent could internally invoke a data query tool. This taught us the value of modular design: each agent has a focus, but they should be able to reuse each other’s capabilities. We had to be careful to avoid circular dependencies though (e.g., not letting agents call each other endlessly). Generally, the orchestrator manages any multi-agent interplay by orchestrating sequential calls (first get data, then feed to next agent).

 With the three main use case agents covered, we now move on to how we evaluate the performance of this multi-agent system and ensure it’s delivering value.

 ### 3. Evaluation Metrics

Implementing a solution is only half the battle; we need to measure its effectiveness. We established a framework for evaluation on two levels: Model performance metrics (how well the AI is answering questions) and Solution impact metrics (the business and user experience outcomes). This combination ensures we don’t just have a clever system, but one that genuinely reduces downtime and is adopted by users.

![image](https://github.com/user-attachments/assets/ac865af4-65b2-4274-9ac9-d0ffd9154491)

#### 3A. Model Performance Metrics (Accuracy & Speed)

To quantitatively evaluate how well the AI agents perform their tasks, we use a mix of automated metrics and targeted testing:

**Accuracy of Answers:**
* For the RAG agent, we leverage **RAGAS (Retrieval-Augmented Generation Assessment Suite)** to evaluate answer quality. RAGAS provides metrics like Context Precision and Context Recall which measure whether the retrieved documents were relevant and sufficiently comprehensive. For example, we computed that for a set of test troubleshooting questions, the agent achieved a context precision of ~0.8 (80% of the content it used was relevant) and context recall of ~0.9 (it found 90% of the relevant info available). High scores here indicate the agent is finding and using the right manual pages. We also looked at Answer Correctness in RAGAS – essentially whether the answer is factually consistent with the sources – aiming for >90% correctness in our testing.
* For the SQL and CPLEX agents, we created **test question sets with known answers (ground truth)**. For SQL, this was straightforward: a list of sample queries (like “total downtime last month for Factory X”) where we had the ground-truth from the data warehouse. We then compared the assistant’s answer to the expected answer. Exact match metrics (like exact string match) weren’t suitable because the assistant might phrase things differently. Instead, we computed a **cosine similarity** between the embedding of the assistant’s answer and the embedding of the reference answer. If the answer contains the same facts, this similarity is usually very high (we observed >0.95 for correct answers, whereas incorrect or incomplete answers yielded lower scores). We treated a similarity above a threshold as a success. The SQL agent achieved about 88% success on a 50-question test set initially, and after prompt tweaking, around 95%.
* For the CPLEX agent, evaluating accuracy is trickier since many questions don’t have a single “correct” answer (they depend on assumptions). Our approach was to verify the **internal consistency** of answers. For instance, if the agent claims a 150 unit increase, we check that against the input data. We also had a few known scenario outcomes to test (like a small optimization problem where we manually computed the solution to see if the agent’s solution matched). The agent was consistent in calculations in our tests (no math errors on the test set of scenarios), and when it wrote optimization code, we checked that the code’s output matched the answer given.

**Latency and Throughput:**
* We measure the **average latency per query** and the breakdown by agent. On average, the orchestrator + agent pipeline returned answers in about 2.5 seconds. The SQL agent was fastest (~2 seconds, most of which was DB query time for large results), the RAG agent was ~3 seconds (embedding + LLM), and the CPLEX agent varied (~3-4 seconds if simple calc, up to 6-8 seconds if it had to solve an optimization or run code). These were within our acceptable range for an interactive system. We log the time taken at each step (e.g., time to retrieve documents, time spent in the LLM call, etc.) for monitoring
* We implemented **asynchronous** calls where possible. For example, while the LLM is thinking, in some cases we could prefetch data in parallel. Also, retrieving multiple documents from the vector DB is parallelized. This shaved off some latency. We enforce an SLA (Service Level Agreement) where if an agent takes longer than, say, 10 seconds, the orchestrator will time out and return a fallback message (“I’m still working on that, please wait…” or an apology and partial info). In practice, we rarely hit the timeout except in some heavy optimization queries, which we are optimizing.
* **Token usage and cost metrics:** For each response, we record the model name used, tokens in/out, and estimate cost. For instance, a typical SQL agent answer might use ~1500 tokens (prompt + answer) on GPT-4, which costs around $0.09 in OpenAI API usage. RAG answers were similar size. The CPLEX answers including code were larger, ~1800 tokens ($0.11). We track these to ensure the service stays within budget. We also log the distribution of model calls (if we ever route some requests to GPT-3.5 vs GPT-4 to save cost for simpler queries).

**Conversation Quality (Human Evaluation):** Beyond automated metrics, we periodically do human-in-the-loop evaluation. Domain experts review a sample of Q&A transcripts, scoring them on correctness, clarity, and utility. This qualitative feedback helped us fine-tune prompts and decide on things like how much detail to include. Over time, the scores improved as we addressed issues (for example, early on the RAG agent sometimes gave generic advice lacking specific detail – human reviewers caught that, we then adjusted the retrieval to include more specific text). 

#### 3B. Solution Impact Metrics (User Engagement & Business Outcomes)
Ultimately, our success is measured by how much we reduce downtime and how well the solution is used by factory staff. 

**Hypothesis for the Solution:** Using LLM Application would reduce the triage time during unplanned downtime at Factories. 

To test the above hypothesis, we defined several key **solution-level metrics:**

**User Engagement:**
*	*Daily Active Users (DAU):* How many distinct users engage with the assistant per day. We saw steady growth in DAU as we rolled out to more factories. For instance, after launch in Factory Lincoln and two others, we had  close to 20 users/day; after 6 weeks and adding 5 more sites, it grew to close to 50 DAU. Weekly active users reached around 120, indicating some users use it a few times a week rather than daily.
*	*Average Conversations per Day:* We track total queries per day. In the first month, it averaged ~100 queries/day and by third month close to 200 queries/day as trust in the tool grew. Peaks aligned with shift changes (morning meetings and end of day reports).
*	*Average Questions per Session:* On average, users asked 3.2 questions per session. This tells us users often follow up an answer with further queries (which is exactly what we want – it indicates dialogue). A session often started with a data question, followed by a troubleshooting question, etc. If this number was 1.0 (single question and done), it might mean they aren’t engaging deeply. 3+ is a healthy sign of interaction. Our target is to keep this around 3-5, meaning users find enough value to continue the conversation a bit.
*	*Recommendation Click-through Rate:* We offer suggested questions; about 30% of sessions had the user click on a recommended follow-up. This is a decent engagement rate, showing that the AI’s suggestions are relevant. It also reduces effort for the user to think of the next question.

**Downtime Reduction:**
*	*Average Downtime per Factory:* This is the core business metric we want to impact. We measure the hours of downtime per factory per week. Prior to the assistant,  Factory Lincoln averaged 10 hours/week of down time. After deploying the assistant (and getting people to use it when issues arise), we observed a downward trend – over 3 months, Lincoln averaged ~8.5 hours/week, a 20% reduction. While many factors influence downtime, we did a comparative study: factories using the assistant saw an average 5-10% larger drop in downtime than those not using. This suggests a positive correlation.
* *Uplift Correlation with Engagement:* We actually calculated the correlation between the number of queries users made and the reduction in downtime in subsequent weeks. There was a moderate **negative correlation (around -0.5)** – meaning more questions to the assistant tended to coincide with lower downtime. Of course, correlation isn’t causation, but in feedback interviews, engineers reported that the assistant helped them identify issues faster and fix problems more promptly, contributing to these gains.
* *Resolved Issues and Escalation Rate:* We track if a conversation led to an action like a maintenance ticket or a schedule change. Approximately 40% of troubleshooting questions (like the jam example) resulted in the user either implementing the fix directly or creating a follow-up action through the system. The remaining 60% were either informational or the user decided to escalate to a human expert. An **escalation rate of 60%** might sound high, but considering these are often complex issues, it’s actually good that the assistant handled 40% without further help. Our goal is to improve this by expanding the knowledge base so more questions can be fully answered by the assistant.

**User Satisfaction and Adoption:**
*	*Feedback Scores:* We added a simple feedback prompt after an answer (“Was this answer helpful? (thumbs up/down)”). We’re seeing about 85% thumbs-up across all interactions. Thumbs-down often come when the assistant couldn’t answer due to lack of data or gave a too-generic response. Each thumbs-down is reviewed to classify the cause (e.g., retrieval failure, formatting issue, etc.) and we address them in weekly model update meetings.
*	*Training and Onboarding:* An indirect metric – how quickly new users pick up the tool. We found that after a short demo, most users needed minimal training to start asking questions. If lots of clarification or support were needed, that would indicate the interface or responses were confusing. So far, adoption has been smooth, evidenced by increasing usage stats.

Below is a summary table of some key metrics with their definitions and sample values from our deployment.

![Screen Shot 2025-05-11 at 9 13 01 PM](https://github.com/user-attachments/assets/c6d1a2a8-e243-4fb5-9040-3148ff8c5277)

These metrics are logged during the conversations and continuously tracked in our analytics dashboard. They help us identify where to improve in production: e.g., if context recall drops, maybe we need to add more documents or improve search; if latency spikes, investigate which agent is slow, etc. And importantly, they show the value to stakeholders (e.g., “We cut downtime by 30%, translating to $X saved per week”). 

### 4.Cost Implementation
Building and running an AI multi-agent system incurs costs which we carefully estimated and optimized. Costs come from two main sources: infrastructure (fixed costs) and API/model usage (variable costs). Below we break down the costs for each use case/agent per query, and then discuss how we manage fixed vs variable costs and strategies to keep the operations economical.

#### 4A. Per-Query Function Call Breakdown by Use Case

![Screen Shot 2025-05-11 at 9 20 16 PM](https://github.com/user-attachments/assets/476c09ac-2a8f-46a8-8de3-488dfadaf7f0)

![Screen Shot 2025-05-11 at 9 20 36 PM](https://github.com/user-attachments/assets/fa85f17c-5bac-41d2-9de9-748e65dfa06b)


#### 4B. Cost Optimization Strategies
To ensure the solution remains cost-effective as it scales, we implemented several optimizations and best practices:
*	**Model Tiering:** Not every query needs the most expensive model. We use GPT-o3,4 for its reliability and quality, but we tested GPT-4.1 for simpler tasks. For example, generating straightforward SQL or doing basic math can often be handled by GPT-4.1 at 1/15th the cost. This dynamic model selection saved about 30% in cost in our load tests, with negligible impact on answer quality for those targeted parts. The orchestrator was trained on understanding which questions were hard or easy based on the examples we had provided in the prompt.
*	**Prompt Token Pruning:** We noticed some prompts had a lot of system text or example shots that weren’t always needed. We optimize prompts by:
  *	Shortening instructions (without losing meaning). E.g., we removed redundant sentences from the system message over time.
  *	Using tokens frugally in examples (e.g., short placeholder names instead of long ones in sample SQL).
  *	Carrying forward only relevant context in multi-turn chats. Instead of sending the entire conversation history to the model, we distill it. For instance, for Q4 we didn’t resend the full table from Q1, we just sent the key numbers needed. This reduced prompt size dramatically. We essentially compress context: the orchestrator maintains a state (like context = {factory: Lincoln, last_week_top_events: [...]}) and can regenerate a brief summary of it to feed into the prompt, rather than raw past dialogue. This ensures the token count stays low as conversations grow.
* **Caching Results:** We implemented caching at multiple levels:
* *Embedding Cache:* Many users might ask similar questions (e.g., “top 10 stoppages last week” for different factories, or even the same). We cache embeddings for frequent queries and even the retrieval results. If an identical query embedding is seen, we reuse the stored similar docs instead of recomputing. This saves time and a tiny bit of cost.
* *Answer Cache:* For certain repetitive questions (especially ones that don’t change often, like “What’s the procedure for X?”), we cache the final answer. If the same user asks again or another user asks the identical thing, we can directly return the cached answer (maybe with a note that it’s using cached data if freshness matters). We do this carefully for questions that are by nature static (manual instructions don’t change day to day). For dynamic queries (like anything time-bound or data-bound), we avoid caching because we want fresh data.
* *Partial computation cache:* In our scenario analysis agent, if a similar simulation was run before, we cache the result or the generated code. For example, if two users both ask about Maggi on conv2 vs conv3 in the same week, the second time we can skip recalculating since we know the result (assuming conditions unchanged). This didn’t happen often yet, but as scenarios repeat, it could help.
*	**Batching and Parallelization:** Where possible, we batch multiple tasks into one API call. For instance, if we need embeddings for multiple pieces of text (like indexing a manual), we send a batch to the embedding API (OpenAI allows up to 16 texts in one request) – this is more efficient than calling one by one. Similarly, for generating multiple follow-up questions, we ask the model to output a list of 3 suggestions in one go, rather than separate calls for each suggestion.
*	**Cost Visibility:** We built cost logging into our observability. Every answer log includes an estimated cost field. This transparency means at any point we can sum up and attribute cost to features (e.g., see monthly cost of SQL agent vs RAG agent). If one is disproportionately high, we drill down and optimize that part. This also helps in justifying the spend vs the savings achieved – we can show, for example, that a month of answers cost $500 while saving an estimated $50,000 in downtime. That ROI makes the cost easily acceptable to management, but keeping tabs on it ensures it doesn’t creep into something surprising.

By applying these strategies, we’ve kept the operational cost well within acceptable limits. As we plan to scale to more factories and perhaps 10x the queries, these optimizations will allow us to handle the load without a linear 10x cost increase (ideally, we grow usage while maybe only 5x cost, thanks to efficiencies)

### 5.Production Considerations
Moving from a successful prototype to a production system required careful planning around success criteria, deployment, maintenance, and continuous improvement. Here we outline key considerations and best practices we adopted to ensure the system is robust and delivers ongoing value in a real factory setting.

##### 5A. Success Criteria and KPI Targets
Before broad rollout, we defined what success looks like. The evaluation metrics discussed serve as our KPIs:
*	**Accuracy and Reliability:** We set a target that at least 90% of user queries should be answered usefully without human intervention. “Usefully” means either giving the user the info they needed or appropriately saying it doesn’t know (but not giving wrong info). We monitor the thumbs-up rate and aim to minimize the critical failure cases (like confidently incorrect answers).
*	**Downtime Reduction:** The ultimate business success criterion is reduction in downtime. We targeted a 10% reduction in the first 3 months at pilot sites. We surpassed this in Lincoln factory (30%), which gave confidence. We also look at the number of incidents resolved faster. If previously a certain class of problem took, say, 30 minutes for an engineer to diagnose via manual logs but now they get the answer in 5 minutes via the assistant, that’s a win (even if not directly counted in downtime, it’s efficiency).
*	**User Adoption and Satisfaction:** We consider the tool successful if it becomes a go-to resource for engineers and operators. If after 6 months, it’s gathering dust (low usage), then it failed to integrate into workflows. So far, adoption is good. We set a goal of at least 50% of shift leads using it weekly. We are tracking towards that. High satisfaction (via feedback) is another goal; any persistent pattern of dissatisfaction triggers a review and fixes.
* **Scalability:** A criterion for success is that the system can scale to all global factories (say 50 sites, 500 users) without degradation. This means the architecture must handle concurrent conversations, the latency should remain low, and costs per user remain sustainable. We tested the system under higher loads (simulate many parallel queries) to ensure the orchestrator and agents can scale horizontally. Success is when adding more users simply means scaling out pods or instances, with linear cost, and no major refactoring needed.
*	**Compliance and Security:** Though not explicitly mentioned earlier, in production we must meet IT security requirements. Success meant passing security review: ensuring no sensitive data leakage, using encryption for data in transit, proper authentication (only authorized employees can access the system), and logging for audit. We integrated with the company’s SSO for authentication in the web UI, and limited the system to the internal network. We also have content filtering (via guardrails) to avoid any inappropriate use (like someone trying to get the AI to do something outside scope).

##### 5B. Deployment Architecture (Microservices vs. Agents SDK)
We chose a **microservices architecture** to deploy the solution, aligning with the diagram presented:
*	The **Web Layer (React UI + Auth)** is one service. It handles the interface and user management.
*	The **Orchestrator Service** is a central microservice that receives questions from the UI (via an API call) and coordinates with agents. It’s stateless (other than pulling context from Redis), which means we can run multiple instances behind a load balancer to handle many users.
*	**The Agent Services:** We have logically separated agents (SQL Agent, RAG Agent, CPLEX Agent, Report Agent) as depicted. During development, we implemented them in one codebase for simplicity, but for production we split them for scalability and isolation. Each agent could be a separate microservice with its own API that the orchestrator calls (e.g., POST /sql-agent with the query, returns the result). This way, each can scale or be updated independently. If one agent fails or needs a restart (e.g., maybe the CPLEX agent has heavier memory usage), it doesn’t bring down the whole system.
*	**Why microservices:** This decoupling follows the single responsibility principle – each agent focuses on its domain. It also allowed different teams or developers to work on agents in parallel. For instance, if we want to improve the RAG agent, we deploy a new version of that service without touching the others. It also means we could theoretically rewrite one agent in a different language or framework if needed, as long as it communicates over the agreed interface (usually REST or gRPC).
*	**Agents SDK alternative:** We considered using an Agents SDK or framework (like Semantic Kernel SDK, or Azure’s OpenAI Function Calling with a single prompt that can call tools). In a lab environment, those are great for rapid prototyping (you can have one process that uses a chain-of-thought and functions to do everything). However, for production and maintainability, we found the explicit microservice approach clearer. It’s easier to test each agent in isolation and stub them out in case of issues. That said, we did borrow ideas from such SDKs. For example, the orchestrator’s logic is akin to an agent executor deciding which tool to use, and the guardrails service is similar to hooking in an output parser or validator. We essentially implemented a custom version of an agent framework tailored to our exact needs and integrated with our systems. This gave us more control over error handling and logging.
*	**Containerization and Orchestration:** Each service (or set of related services) is containerized (Docker). We use Kubernetes to orchestrate them, which gives resiliency (auto-restarting crashed pods, scaling out, etc.). The vector database is also running in the cluster as a stateful service. The SQL DB and ServiceNow integration are external and done via Azure endpoints.
*	**Deployment environments:** We maintain separate environments (dev, staging, prod). We test new versions of agents in staging with sample queries and some live shadow traffic before promoting to prod. CI/CD pipelines automate this, ensuring that any code change goes through linting, automated tests (we wrote tests for prompt outputs on known inputs), and then deployment.

##### 5C. Model Versioning and Switching

LLM technology evolves quickly, so we planned for model version management:
*	We parameterize the model names (and API keys) in configuration. That means switching from one model to another is as simple as a config change and restart, no code change needed. For example, GPT4_MODEL = "gpt-4-0314" could be updated to a new version gpt-4-2025 when available.
*	We keep track of which model was used for each conversation in the logs. If we upgrade the model and suddenly something regresses, we can pinpoint that it might be due to the new model.
*	The system is designed to be model-agnostic in logic. If an open-source model were to be used in the future (hosted on a GPU server we control), we’d implement a wrapper so that from the agent’s perspective it’s just calling a model endpoint with the same interface. This ensures we’re not locked in to one provider.
*	**A/B Testing Models:** We sometimes run two models in parallel to compare. For a subset of queries, the orchestrator can route to an alternate model (say GPT-3.5 vs GPT-4) and we compare outcomes via metrics like user feedback or success on test questions. This helps in justifying the cost of more expensive models or proving that a cheaper model is sufficient for some tasks. We did this when experimenting with GPT-3.5: 50% of users unknowingly got GPT-3.5 for SQL generation, and we monitored if their query success rate differed. It was slightly lower, which led us to mostly use GPT-4 but with some optimizations as mentioned.
*	We maintain **prompt versioning** as well. Prompt tweaks can drastically change outputs. We label our prompts with versions (especially for complex ones like the CPLEX agent’s prompt with examples). If we update a prompt, we run our evaluation set to ensure it didn’t break something that was working. Essentially, we treat prompts like code – version controlled and tested. You can do it via custom functions or via a 3rd party platform such as IBM Watsonx.gov which is observerability tool with Python SDK for developers that can be easily integration within your system. 
*	**Model updates from provider:** When OpenAI or others update model behaviors, we have to be vigilant. For example, if a new GPT-4 patch makes it follow formatting instructions differently, our table output might change. To handle this, we pin to specific model snapshots (like gpt-4-0314 which is the March 2024 version) and only move to newer ones after testing. This way, we’re not surprised by silent changes in model behavior.

##### 5D. Performance and Latency Optimizations in Production
Latency can be a make-or-break factor for user experience. We implemented several tactics to keep the system snappy:
*	**Parallel Agent Actions:** The orchestrator, in some cases, can call multiple things at once. For example, when the CPLEX agent is invoked for a scenario, in parallel it might fetch data from SQL and call the LLM -> here the microservices oriented architecture was useful to subdivide and parallelize the calls but it’s possible to use Function calls to implement this design as well with some custom coding. 
*	**Profiling and reducing overhead:** We profiled each step. For instance, if the vector search took 500ms, can we reduce vector dimension or optimize indexing? We did fine-tune some of these. We also ensure our code doesn’t do unnecessary serialization/deserialization; passing data between services is done in a lightweight JSON format.
*	**SLA enforcement and fallbacks:** As mentioned, we set timeouts. If an agent doesn’t respond in time, the orchestrator has fallback logic. For example, if the CPLEX agent times out (maybe the optimization took too long), the assistant will apologize and offer to email the results later, or it might present whatever partial info it has. For RAG, if the LLM is taking unusually long (perhaps waiting on the API), we might at least say “Still looking up the manual…” to the user. These little touches keep the user informed rather than staring at a blank screen.
*	**Warm-up and caching:** The first call to an LLM model might be slow (cold start at LLM API’s end). We mitigate this by a warm-up call after deployment (ping each model once with a trivial prompt so it’s loaded). Also, our caching mentioned earlier not only saves cost but also latency (returning a cached answer is virtually instant). We do that for known frequently repeated queries, which users appreciate (they don’t even know it’s cached, it just feels super fast).
*	**Scaling horizontally:** If we notice increased latency due to heavy load (many simultaneous users), we have auto-scaling rules to spawn more instances of services. The stateless nature of orchestrator and agents allows this. We simulate load to ensure the system can scale to, say, 100 concurrent conversations without slowing down during stress testing. The vector DB and SQL DB are sized and replicated to handle concurrent queries as well.

##### 5E. Observability and Logging
Observability is crucial for a complex AI system to troubleshoot issues and ensure reliability:
* **Centralized Logging:** During user testing phase, every request and response, along with meta-data (timestamps, model used, tokens used, any errors), is logged to a centralized log system. We use unique request IDs to trace a question through the orchestrator to an agent and back. This way, if a user reports a weird answer, we can find the exact conversation in logs and see what happened (including the full prompt that was sent to the model). Sensitive data in prompts (like actual table data) is handled carefully (we may mask certain fields in logs if needed for privacy). For normal production loads, we chose to log random user conversation (10-20%) due to storage and cost constraints. 
* **Analytics Dashboard:** We built a small internal dashboard that surfaces the metrics discussed (like usage, accuracy, cost) using Watsonx.gov and has default and custom views. It pulls from the logs and database of conversations. This dashboard is reviewed in our weekly meetings to spot trends. It helps us answer questions like: Did the latest deployment improve things or cause any drop in quality? Are there new types of questions coming in that we didn’t anticipate?
•	**Guardrails and Monitoring for Content:** While our domain is mostly technical, we still implemented guardrails to ensure the AI doesn’t produce inappropriate content or leak information. The Guardrails Service scans answers for things like profanity (shouldn’t happen with our prompts, but just in case) or hallucinations (e.g., it might detect if an answer has a citation that wasn’t in the retrieved text). We utilized the watsonx.gov python native functions to achieve this as well as some custom ones for checking SQL & CPLEX code syntax and validation. If something is flagged, we log it and modify the response or route it for human review. In practice, we rarely hit these triggers, but it’s good to have a safety net.
*	**User Feedback Loop:** The thumbs-up/down feedback is also logged and aggregated. If certain queries often get thumbs-down, we prioritize them for investigation. We sometimes follow up directly with users (since it’s internal) to ask what went wrong. This direct feedback loop is gold for iterative improvement.

### 6.Conclusion and Future Enhancements
In summary, taking this multi-agent LLM solution to production involved not just writing code, but setting up a whole ecosystem for it to thrive: defining what success means, deploying it in a robust architecture, keeping a close eye on it through monitoring, and continuously refining it through feedback and testing. By following these practices, we’ve achieved a reliable AI assistant that is now an integral tool in reducing downtime and improving operations in our manufacturing processes. The journey doesn’t end here – we will keep learning and improving the system, but the cookbook above provides a solid foundation for anyone looking to implement a similar Generative AI multi-agent solution in an enterprise setting.
Looking ahead, our modular architecture and use-case-driven agent design open up a host of technology-forward enhancements:
* **1.	Knowledge-Graph-Enhanced Retrieval**
Integrate a lightweight manufacturing knowledge graph (e.g. Neo4j with OPC UA ontologies) alongside vector search. Graph queries can surface relationships—“all machines that share a part with Conveyor 3”—to boost retrieval precision and reduce redundant API calls by combining semantic search and graph traversals in a single pipeline.
* **2.	Cost-Aware Model Orchestration**
Embed dynamic model selection logic in the Orchestrator: route demanding reasoning tasks to GPT-4o or GPT-4.1 low-latency summarizations to GPT-o3/o4, and factual lookups to fine-tuned open-source embeddings. By tracking per-call cost and performance in real time, the system can automatically balance accuracy vs. spend, keeping monthly API budgets within forecasted targets without manual overrides.
* **3.	Service-Fusion to Reduce Chattiness**
Fuse multiple data-fetching steps into single orchestrator calls wherever possible. For instance, bundle a vector search + SQL lookup + ServiceNow ticket check into one composite function-call, reducing back-and-forth latency, lowering total token usage, and simplifying observability by collapsing multiple logs into one trace.
* **4.	Real-Time Sensor-Driven Simulation**
Augment the CPLEX Agent’s simulation inputs with live IoT data streams (via Azure Event Hubs or Kafka) to run “what-if” scenarios on fresh sensor readings. Combine these with short-term forecasting models (e.g. LSTM on temperature or vibration data) to predict imminent downtime—and recommend preventive actions before a stoppage occurs.
* **5.	Self-Optimizing Agents via Reinforcement Signals**
Introduce a lightweight reinforcement-learning layer or bandit setup that rewards agents for outcomes: e.g., a successful fix within 30 minutes or a supervisor’s “approve” click. Use these signals to fine-tune prompts, update retrieval filters, or adjust decision thresholds—so the agents continually optimize for real-world efficacy.
* **6.	Cross-Process Expansion & Global Rollout**
Leverage the same multi-agent pattern for other domains—quality-control root-cause analysis, supply-chain bottleneck forecasting, or energy-usage optimization. Swap in new data connectors (SAP, MES), curate relevant manuals, and reuse the orchestration, memory, and monitoring layers to scale rapidly across sites worldwide.

#### 7.	Reference cookbooks and Resources
* [Open AI model selection guide](https://cookbook.openai.com/examples/partners/model_selection_guide/model_selection_guide#3a-use-case-long-context-rag-for-legal-qa) – very detailed analysis on the latest open ai models applied to various use cases along with performance and cost considerations. 
* [GPT-4.1 Prompting Guide] (https://cookbook.openai.com/examples/gpt4-1_prompting_guide) – detailed prompting techniques including structured multi-shot prompts for accuracy and completeness. 
*	[Watsonx.gov guide](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-python-lib.html?context=wx) – for creating model and solution metrics which helps with performance logging and observability of the platform. 
* [Agent Building](https://platform.openai.com/docs/guides/agents) – building agents including function, tool and guardrail calls. While this SDK not exactly used, the structure of the agent was used in the microservices architecture. 
* [RAG with OpenAI & Azure Cognitive Search](https://docs.microsoft.com/azure/search/cognitive-search-vector-search) - Step-by-step on indexing documents in Azure Cognitive Search, using OpenAI embeddings, and building a retrieval-augmented agent.
* [OpenAI Assistants API Tool Orchestration](https://cookbook.openai.com/examples/responses_api/responses_api_tool_orchestration) - Example of combining function-calling, file search, and multi-turn orchestration in a single assistive agent.
* [Azure AI Semantic Kernel](https://github.com/microsoft/semantic-kernel) - Open-source SDK for orchestrating LLMs, retrieval plugins, memory, and planner agents on Azure.
* [Python Docplex API Reference](https://ibmdecisionoptimization.github.io/docplex-doc/) - Python library for building and solving optimization problems with CPLEX, including LP/MIP examples.
* [RAGAS (Retrieval-Augmented Generation Assessment Suite)](https://github.com/prmsllr/ragas) - Open-source toolkit for evaluating RAG systems on precision, recall, and answer faithfulness.




























 


















 









































