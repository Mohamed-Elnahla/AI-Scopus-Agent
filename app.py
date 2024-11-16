import streamlit as st
import os
import requests
import textwrap
import json

# Import necessary modules from your original script
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Set up API keys from Streamlit secrets
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["SCOPUS_API_KEY"] = st.secrets["SCOPUS_API_KEY"]

# Define the scopus_search tool
@tool
def scopus_search(query: str, Count:int) -> str:
    """
    Search for publications in Scopus matching the query and include abstracts.

    Parameters:
    - query (str): A Scopus-formatted query string.
    - Count (int): Number of results to return by default 10 if not stated
    Returns:
    - str: A formatted string of search results, including titles, authors, journals, publication years, abstracts, and links.

    Query Construction Guidelines:

    1. Basic Topic Search:
       - Purpose: Find publications related to a specific topic.
       - Syntax: TITLE-ABS-KEY("search terms")
       - Example: TITLE-ABS-KEY("machine learning")

    2. Author Search:
       - Purpose: Retrieve works by a specific author.
       - Syntax: AUTHLASTNAME("author's last name") or AUTH("author's full name")
       - Example: AUTHLASTNAME("Smith") or AUTH("Smith, John")

    3. Journal Search:
       - Purpose: Locate articles published in a particular journal.
       - Syntax: SRCTITLE("journal name")
       - Example: SRCTITLE("Journal of Artificial Intelligence Research")

    4. Year Filter:
       - Purpose: Filter publications by publication year.
       - Syntax: PUBYEAR = year, PUBYEAR > year, or PUBYEAR < year
       - Example: PUBYEAR > 2015

    5. Combination of Criteria:
       - Purpose: Combine multiple search criteria.
       - Syntax: Use logical operators (AND, OR, NOT) and parentheses to group terms.
       - Example: TITLE-ABS-KEY("deep learning") AND PUBYEAR > 2018

    6. Exact Phrase Search:
       - Purpose: Search for an exact phrase within a specific field.
       - Syntax: FIELD("exact phrase")
       - Example: TITLE("convolutional neural network")

    7. Subject Area Search:
       - Purpose: Find publications within a specific subject area.
       - Syntax: SUBJAREA("subject area")
       - Example: SUBJAREA("computer science")

    8. DOI Search:
       - Purpose: Retrieve a publication using its DOI.
       - Syntax: DOI("doi number")
       - Example: DOI("10.1016/j.artint.2020.103536")

    9. Affiliation Search:
       - Purpose: Find publications associated with a specific institution.
       - Syntax: AFFIL("institution name")
       - Example: AFFIL("Massachusetts Institute of Technology")

    10. Open Access Filter:
        - Purpose: Limit results to open-access publications.
        - Syntax: OA
        - Example: TITLE-ABS-KEY("quantum computing") AND OA

    11. Language Filter:
        - Purpose: Restrict results to publications in a specific language.
        - Syntax: LANGUAGE("language")
        - Example: TITLE-ABS-KEY("renewable energy") AND LANGUAGE("English")

    Additional Notes:
    - Logical Operators: Use AND, OR, and NOT to combine search terms.
    - Grouping: Use parentheses to group terms and control the logic of the search.
    - Wildcards: Use * to represent any group of characters and ? for a single character.
    - Field Codes: Common field codes include TITLE, ABS (abstract), KEY (keywords), and AUTH (author).

    By following these guidelines, you can construct effective queries to retrieve relevant publications from Scopus using the scopus_search function.
    """

    api_key = os.getenv("SCOPUS_API_KEY")
    if not api_key:
        return "Scopus API key is not set."

    url = "https://api.elsevier.com/content/search/scopus"
    params = {
        "query": query,
        "count": Count,  # Number of results to retrieve
        "sort": "relevance",
        "start": 0    # Start index
    }
    headers = {
        "X-ELS-APIKey": api_key,
        "Accept": "application/json"
    }
    response = requests.get(url, params=params, headers=headers)
    if response.status_code != 200:
        return f"Error: {response.status_code} - {response.text}"

    data = response.json()
    entries = data.get("search-results", {}).get("entry", [])
    if not entries:
        return "No results found."

    # Format the first few results with abstracts
    formatted_results = []
    for entry in entries:
        title = entry.get("dc:title", "No title")
        authors = entry.get("dc:creator", "No authors")
        journal = entry.get("prism:publicationName", "No journal")
        year = entry.get("prism:coverDate", "No date")[:4]
        doi = entry.get("prism:doi", "DOI not available")
        doi_url = f"https://doi.org/{doi}" if doi != "DOI not available" else "DOI not available"
        link = entry.get("prism:url", "No link")

        # Abstract retrieval and formatting
        abstract = "No abstract available."
        abstract_link = next(
            (link_item["@href"] for link_item in entry.get("link", []) if link_item["@ref"] == "self"),
            None
        )

        # Try to fetch the abstract from Scopus
        if abstract_link:
            try:
                detail_response = requests.get(abstract_link, headers=headers)
                if detail_response.status_code == 200:
                    detail_data = detail_response.json()
                    raw_abstract = detail_data.get("coredata", {}).get("dc:description", "No abstract available.")
                    # Format the abstract to clean unnecessary spacing and wrap text
                    abstract = "\n".join(textwrap.wrap(raw_abstract.strip(), width=80))
                else:
                    abstract = f"Error fetching abstract from Scopus: {detail_response.status_code}"
            except requests.exceptions.RequestException as e:
                abstract = f"Error fetching abstract from Scopus: {e}"

        # If abstract is still unavailable, try ScienceDirect
        if abstract == "No abstract available." and doi != "DOI not available":
            science_direct_url = f"https://api.elsevier.com/content/article/doi/{doi}"
            try:
                sd_response = requests.get(science_direct_url, headers=headers)
                if sd_response.status_code == 200:
                    sd_data = sd_response.json()
                    raw_abstract = sd_data.get("full-text-retrieval-response", {}).get("coredata", {}).get("dc:description", "No abstract available.")
                    # Format the abstract to clean unnecessary spacing and wrap text
                    abstract = "\n".join(textwrap.wrap(raw_abstract.strip(), width=80))
                else:
                    abstract = f"Error fetching abstract from ScienceDirect: {sd_response.status_code}"
            except requests.exceptions.RequestException as e:
                abstract = f"Error fetching abstract from ScienceDirect: {e}"

        # Format the result as a dictionary
        formatted_results.append({
            "Title": title,
            "Authors": authors,
            "Journal": journal,
            "Year": year,
            "DOI": doi,
            "DOI_URL": doi_url,
            "Abstract": abstract,
            "Link": link
        })


    # Convert formatted results to JSON
    return json.dumps(formatted_results, indent=4)

# Define the summarize_results function
def summarize_results(results: str) -> str:
    """
    Summarize the search results using the Llama-3-Groq model.

    Parameters:
    results (str): Raw search results as input.

    Returns:
    str: A formatted and readable summary of the results.
    """
    # Prepare the prompt with the input results
    prompt = f"Summarize the following search results:\n\n{results.strip()}"
    
    # Invoke the LLM with the constructed prompt
    response = llm_with_tools.invoke(prompt)
    
    # Ensure the content is clean and formatted for readability
    formatted_response = response.content
    
    return formatted_response

# Process user input and display results
def process_user_input(user_input):
    response = chain.invoke({"input": user_input})

    # Process the model's response
    if response.tool_calls:
        st.write("Processing tool calls...")
        all_results = []

        for tool_call in response.tool_calls:
            tool_name = tool_call['name']
            tool_args = tool_call['args']

            # Display tool name and arguments
            st.markdown(f"**Tool Name:** {tool_name}")
            st.markdown(f"**Tool Arguments:** {tool_args}")

            # Execute the tool with the provided arguments
            results = scopus_search(tool_args)
            if isinstance(results, str):
                try:
                    # Parse the string result into JSON
                    results = json.loads(results)
                except json.JSONDecodeError:
                    # Handle cases where the result is not valid JSON
                    st.warning(f"Warning: Failed to parse result as JSON: {results}")
                    continue  # Skip to the next result

            all_results.extend(results)

        if all_results:
            st.markdown("### Detailed Results")

            # Loop through each result and format as Markdown
            for result in all_results:
                st.markdown(
                    f"**Title:** {result.get('Title', 'N/A')}\n\n"
                    f"**Authors:** {result.get('Authors', 'N/A')}\n\n"
                    f"**Journal:** {result.get('Journal', 'N/A')} ({result.get('Year', 'N/A')})\n\n"
                    f"**DOI:** {result.get('DOI', 'N/A')} ([Link]({result.get('DOI_URL', '#')}))\n\n"
                    f"**Abstract:** {result.get('Abstract', 'N/A')}\n\n"
                    f"**Link:** [Full Text]({result.get('Link', '#')})\n\n"
                    "---"  # Add a horizontal line between results
                )

        # Summarize results
        str_results = '\n\n'.join([
            '\n'.join([
                f"Title: {r.get('Title', '')}",
                f"Authors: {r.get('Authors', '')}",
                f"Journal: {r.get('Journal', '')} ({r.get('Year', '')})",
                f"DOI: {r.get('DOI', '')}",
                f"DOI URL: {r.get('DOI_URL', '')}",
                f"Abstract: {r.get('Abstract', '')}",
                f"Link: {r.get('Link', '')}"
            ])
            for r in all_results
        ])
        summary = summarize_results(str_results)

        # Display summary
        st.markdown("### Summary of Search Results")
        st.markdown(summary)
    else:
        # If no tool calls, display the content directly
        st.markdown(response.content)

# System message and chain setup
system_message = (
    "You are a research assistant with access to the 'scopus_search' tool, "
    "which retrieves academic publications based on a query. "
    "For any user query that seeks information on academic publications, "
    "use the 'scopus_search' tool to provide accurate and up-to-date information."
)

user_message_template = "{input}"

prompt = ChatPromptTemplate.from_messages([
    ("system", system_message),
    ("user", user_message_template)
])

# Initialize the Llama-3-Groq model
llm = ChatGroq(
    model="llama3-groq-70b-8192-tool-use-preview",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Bind the tools to the model
tools = [scopus_search]
llm_with_tools = llm.bind_tools(tools)

chain = prompt | llm_with_tools

# Streamlit app main function
def main():
    st.title("Scopus Search and Summarize App")

    # Instructions
    st.markdown("""
    Enter your query below to search for academic publications using Scopus and get a summary.
    """)

    # User input
    user_input = st.text_input("Enter your query:")
    if st.button("Search"):
        if not user_input:
            st.warning("Please enter a query.")
        else:
            with st.spinner("Processing..."):
                process_user_input(user_input)

if __name__ == "__main__":
    main()
