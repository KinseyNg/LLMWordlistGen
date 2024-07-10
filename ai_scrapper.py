# Import the required libraries
import streamlit as st
from scrapegraphai.graphs import SmartScraperGraph, SmartScraperMultiGraph
from Wappalyzer import Wappalyzer, WebPage
import pandas as pd
from phi.assistant import Assistant
from phi.tools.hackernews import HackerNews
from phi.llm.openai import OpenAIChat
from urllib.parse import urlparse
import os
import requests
from bs4 import BeautifulSoup
import difflib
from dotenv import load_dotenv
import json
import shutil
import concurrent.futures
from urllib.parse import urlparse, urlunparse
import sqlite3
import time

# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")

PRODUCTION = True
debugging_enabled = False


def fetch_html(url):
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.text
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def extract_html_structure(html):
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(["script", "style"]):
        script.decompose()  # Remove script and style elements
    return str(soup)


def is_similar_html(html1, html2, threshold=0.9):
    seq = difflib.SequenceMatcher(None, html1, html2)
    return seq.ratio() > threshold

def filter_unique_urls(urls, base_domain, similarity_threshold=0.9):
    unique_urls = []
    html_structures = []
    st.write("Checking URLs and removing the URL with similar context")
    for url in urls:
        parsed_url = urlparse(url)
        if parsed_url.netloc != base_domain:
            continue  # Skip URLs that are not in the same domain
        html = fetch_html(url)
        if html is None:
            continue
        
        html_structure = extract_html_structure(html)

        is_unique = True
        for existing_structure in html_structures:
            if is_similar_html(html_structure, existing_structure, threshold=similarity_threshold):
                is_unique = False
                break

        if is_unique:
            unique_urls.append(url)
            html_structures.append(html_structure)
    
    return unique_urls

def clean_url(input_url):
    # Parse the input URL
    parsed_url = urlparse(input_url)
    
    # Extract the scheme and netloc (domain)
    scheme = parsed_url.scheme
    netloc = parsed_url.netloc
    
    # If the scheme or netloc is empty, try extracting them manually
    if not scheme or not netloc:
        match = re.match(r'(http[s]?://)?([^/]+)', input_url)
        if match:
            scheme = match.group(1) if match.group(1) else 'http://'
            netloc = match.group(2)
    
    # Ensure the scheme is 'http'
    scheme = 'http'
    
    # Reconstruct the URL without duplicate domains or schemes
    cleaned_url = urlunparse((scheme, netloc, '', '', '', ''))
    
def check_software_in_json(json_data, software_list):
    software_list_lower = [item.lower() for item in software_list]
    matched_software = {}

    for software, details in json_data.items():
        if software.lower() in software_list_lower:
            matched_software[software] = details
    
    return matched_software

def replace_prompt_reserved_text(prompt, data_list):
    for key, value in data_list.items():
        prompt = prompt.replace("{{"+key+"}}", value)
    return prompt

def read_prompt(mode, type_text, item, source, data_list=[]):

    def run_with_error_handling(content):
        try:
            return hn_assistant.run(content, stream=False)
        except Exception as e:
            print(f"Error occurred during hn_assistant.run: {e}")
            if len(content) > 1:
                mid = len(content) // 2
                part1 = content[:mid]
                part2 = content[mid:]
                result1 = run_with_error_handling(part1)
                result2 = run_with_error_handling(part2)
                combined_result = result1 + result2
                return combined_result
            else:
                return f"Error: Unable to process content - {e}"

    if mode == "exe_one":
        prompt_library_path = os.path.join('./prompt_library', type_text, item)
        try:
            with open(prompt_library_path, 'r') as file:
                file_content = file.read()
            print("File content successfully read and saved to variable.")
        except Exception as e:
            print(f"An error occurred: {e}")
            return None
        
        combined_content = file_content + source
        if debugging_enabled:
            st.write("Input to LLM:", combined_content)
            #st.write("Respose : ", response)
        response = run_with_error_handling(combined_content)

    elif mode == "exe_all":
        prompt_library_path = os.path.join('./prompt_library', type_text.lower())
        st.write(prompt_library_path)
        prompt_list = []
        for root, dirs, files in os.walk(prompt_library_path):
            for file in files:
                with open(os.path.join(root, file), "r") as f:
                    prompt_list.append(f.read())
        if data_list:
            combined_prompts = replace_prompt_reserved_text(prompt_list, data_list)
        else:
            combined_prompts = prompt_list
        st.write(combined_prompts)
        
        response = run_with_error_handling(combined_prompts)

    return response

# Function to clean the summarized result and remove non-JSON parts
def clean_summary(result):
    try:
        # Attempt to parse the result as JSON to ensure it's valid
        json.loads(result)
        return result
    except json.JSONDecodeError:
        # If not valid JSON, try to extract JSON-like parts
        try:
            start_idx = result.index("{")
            end_idx = result.rindex("}") + 1
            json_str = result[start_idx:end_idx]
            json.loads(json_str)  # Validate JSON
            return json_str
        except (ValueError, json.JSONDecodeError):
            return "{}"  # Return empty JSON if all else fails

# Get OpenAI API key from user
openai_access_token = api_key
openai_api_key = openai_access_token

def repair_and_load_json(json_str, min_length=10):
    """
    Attempts to repair an incomplete JSON string by progressively removing characters from the end
    until the string can be successfully decoded or becomes shorter than min_length.
    
    Parameters:
    - json_str: The JSON string to repair and decode.
    - min_length: The minimum length of the string to attempt decoding. Prevents infinite loop.
    
    Returns:
    - A Python object decoded from the repaired JSON string, or None if repair was unsuccessful.
    """
    while len(json_str) > min_length:
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            # Remove the last character and try again
            json_str = json_str[:-1]
    return None  # Return None if unable to repair

def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_access_token)
    return llm(input_text)
# Define a function to read existing URLs from the database
def read_existing_urls(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT url FROM successful_urls')
    successful_urls = [row[0] for row in cursor.fetchall()]
    
    cursor.execute('SELECT url FROM unsuccessful_urls')
    unsuccessful_urls = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    return successful_urls, unsuccessful_urls

# Define a function to update the tried number in the database
def update_tried_number(db_path, url):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        UPDATE successful_urls SET tried_count = tried_count + 1 WHERE url = ?
    ''', (url,))
    
    cursor.execute('''
        UPDATE unsuccessful_urls SET tried_count = tried_count + 1 WHERE url = ?
    ''', (url,))
    
    conn.commit()
    conn.close()

# Define a function to generate more paths using the LLM
def generate_more_paths(llm, existing_urls, related_technologies):
    combined_content = "\n".join(existing_urls) + "\n".join(related_technologies)
    additional_paths = llm(combined_content+"Given the URLs found on the URL, give 50 more suggestions, plaintext list of URL only", stream=False)
    return additional_paths.split()

# Define the maximum number of random wordlists to try
MAX_RANDOM_WORDLIST_TRIES = 4000




# Set up the Streamlit app
st.title("LLM-Based Wordlist Generation Framework")
st.caption("by Kinsey v0.1")

if openai_access_token:
    model = st.radio(
        "Select the model",
        ["gpt-4-turbo", "gpt-4-1106-preview","gpt-4o"],
        index=0,
    )
    hn_assistant = Assistant(
        name="LLM Wordlist Creator",
        llm=OpenAIChat(
            model=model,
            max_tokens=1024,
            temperature=0.5,
            api_key=openai_api_key
        ))    
    graph_config = {
        "llm": {
            "api_key": openai_access_token,
            "model": model,
        },
    }
    
    url = st.text_input("Enter the URL of the website you want to scrape")
    remove_folder = st.checkbox('Remove output folder for the domain, (clear and rescan)')
    scraper_step1 = """
    You are a web elements extractor. Your task is to help extract HTML elements of all hyperlinks with complete HTML components. 
    List only HTTP links and return in a structured format. Also, Extract the JavaScript code and library URL used in the given HTML code. 
    For each found HTML code component, remove all the code that is not related to the main business of the website, such as:
    Advertisement code 
    CSS codes 
    Image related code or other items with similar direction 
    Also, Extract the Javascript AJAX call if any 
    Combine and construct all the results into JSON format. Example:
    {
        "Domain": "www.example.com",
        "Server": "Apache 2.3.4",
        "URL": "[Array of Multiple URLs]",
        "HTMLCodes": "[Array of scraped HTML components]",
        "JavaScript": "[Array of scraped JavaScript components]"
    }
    """
    
    urlinfo = urlparse(url)
    base_domain = urlinfo.netloc
    # Extract the base domain with port if present
    parsed_url = urlparse(url)
    base_domain_with_port = f"{parsed_url.hostname}:{parsed_url.port}" if parsed_url.port else parsed_url.hostname

    smart_scraper_graph = SmartScraperGraph(
        prompt=scraper_step1,
        source=url,
        config=graph_config
    )
    
    if st.button("Start"):
        start_time = time.time()
        newpath = './output/' + urlinfo.netloc 
        # Remove the folder if the checkbox is ticked
        if remove_folder:
            if os.path.exists(newpath):
                shutil.rmtree(newpath)
                st.success(f"Folder {newpath} removed successfully.")
            else:
                st.error(f"Folder {newpath} does not exist.")
        st.write("Step 1 : Identify the web framework and technologies with Wappalyzer")
        webpage = WebPage.new_from_url(url)
        wappalyzer = Wappalyzer.latest()
        wappalyzer_return = wappalyzer.analyze_with_categories(webpage)
        df = pd.DataFrame(wappalyzer_return)
        
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        store_filename = newpath + '/server_info'
        f = open(store_filename, "w")
        f.write(str(wappalyzer_return))
        f.close()

        st.dataframe(df, use_container_width=True)
        st.write("Step 2 : Scrapping the URLs from the website")
        
        result = smart_scraper_graph.run()

        #st.write(result["URL"])

        st.write("Step 3 : Find the unique URL from step 2 output")
        st.write("Filtering unique URLs from the list of URLs extracted from the website")
        #Check if saved unique url file exists, if yes, read the file instead of running the function
        if os.path.exists(f"{newpath}/unique_urls.txt"):
            with open(f"{newpath}/unique_urls.txt", "r") as f:
                filtered_urls = f.read().split()
        else:
        
            filtered_urls = filter_unique_urls(result["URL"], base_domain, 0.5) 
            with open(f"{newpath}/unique_urls.txt", "w") as f:
                for url in filtered_urls:
                    f.write(url + "\n")
        
        #st.write("List of unique URLs")
        #st.write(filtered_urls)

        st.write("Step 4 : Crawl all the discovered URLs and save it HTML contents")
        scrapped_html_path = f'{newpath}/scrapped_html'
        if not os.path.exists(scrapped_html_path):
            os.makedirs(scrapped_html_path)

        for unique_url in filtered_urls:
            html_content = fetch_html(unique_url)
            if html_content:
                url_path = urlparse(unique_url).path.strip("/").replace("/", "_")
                if url_path == "":
                    url_path = "index"
                html_filename = f'{scrapped_html_path}/{url_path}'
                with open(html_filename, "w") as html_file:
                    html_file.write(html_content)
        st.write("Saved all in " + scrapped_html_path)

        st.write("Step 5 : Feature Extractor, Summarizer, and Classifier")
        summarized_results_html_path = f'{newpath}/summarized_results'
        summarized_result = ""
        processed_files = 0
        total_files = sum(len(files) for _, _, files in os.walk(scrapped_html_path))
        my_bar = st.progress(0, text="Analyzing progress")
        for root, dirs, files in os.walk(scrapped_html_path):
            for file in files:
                with open(os.path.join(root, file), "r") as f:
                    st.write("Analyzing : " + file)
                    if PRODUCTION:
                        if not os.path.exists(summarized_results_html_path):
                            summarized_result += read_prompt("exe_one", "basic_web", "1_summarize_theme", f.read(), "")
                            summarized_result_clean = clean_summary(summarized_result)
                            with open(summarized_results_html_path, "w") as f:
                                f.write(summarized_result_clean)
                                st.write("Saved LLM summarizing results in " + summarized_results_html_path)
                        else:
                            summarized_result = open(summarized_results_html_path, 'r').read()
                    else:
                        summarized_result = open(summarized_results_html_path, 'r').read() 
                    
                    processed_files += 1
                    progress = processed_files / total_files
                    my_bar.progress(progress, text=f"Analyzing progress: {int(progress * 100)}%")

        #st.write("Summarized results")
        #st.write(summarized_result)

        st.write("Step 6 : CoT for guessing the development process")
        cot_path = f'{newpath}/cot_results'
        if not os.path.exists(cot_path):
            cot_result = read_prompt("exe_one", "basic_web", "2_cot_misconfig", "Framework found by wappalyzer : " + str(wappalyzer_return) + " Website Description : " + summarized_result, "")  
            cot_result_clean = clean_summary(cot_result)
            with open(cot_path, "w") as f:
                f.write(cot_result_clean)
                st.write("Saved CoT results in " + cot_path)
        else:
            cot_result = open(cot_path, 'r').read()
        #st.write(cot_result)
        
        st.write("Step 7 : Guess URL Structure")
        url_structure_path = f'{newpath}/url_structure_results'
        if not os.path.exists(url_structure_path):
            #read all html and combined it into a variable from scrapped_html folder, with the file name 
            scrapped_html = ""
            for root, dirs, files in os.walk(scrapped_html_path):
                for file in files:
                    with open(os.path.join(root, file), "r") as f:
                        scrapped_html += f.read()

            url_structure_result = read_prompt("exe_one", "basic_web", "4_get_url_structure", "Framework found by wappalyzer : " + str(wappalyzer_return) + " summarized URL Structure : " + ', '.join(result["URL"]) + ", All the HTML files captured : " + scrapped_html, "")  
            # Check if url_structure_result is None before concatenating
            if url_structure_result is not None:
                st.write("url_structure_result : " + url_structure_result)
                url_structure_result_clean = clean_summary(url_structure_result)
                with open(url_structure_path, "w") as f:
                    f.write(url_structure_result_clean)
                    st.write("Saved URL Structure results in " + url_structure_path)
                    
            else:
                st.write("url_structure_result is None")
            
            
        else:
            url_structure_result = open(url_structure_path, 'r').read()
        #st.write(url_structure_result)


        st.write("Step 8 : Run OWASP check")
        owasp_path = f'{newpath}/owasp_results'
        if not os.path.exists(owasp_path):
            owasp_result = read_prompt("exe_one", "basic_web", "3_OWASP", "Framework used by wappalyzer : " + str(wappalyzer_return) + " Website Technologies information : " + cot_result + "summarized information : " + summarized_result_clean, "")  
            owasp_result_clean = clean_summary(owasp_result)
            with open(owasp_path, "w") as f:
                f.write(owasp_result_clean)
                st.write("Saved OWASP check results in " + owasp_path)
        else:
            owasp_result = open(owasp_path, 'r').read()
        #st.write(owasp_result)




        if PRODUCTION:
            with open("./prompt_library/wordlist_creation/1_provide_ideas", 'r') as f:
                wordpressprompt = f.read()
            combined_result = summarized_result + "\n" + cot_result + "\n" + owasp_result
            general_ideas = hn_assistant.run("Return plaintext list of URL only, no description : " + wordpressprompt + combined_result, stream=False)
            #st.write("Generating Ideas")
            #st.write(general_ideas)
            # Store the general ideas result
            with open(f"{newpath}/general_ideas_result.txt", "w") as f:
                #Clean to pure JSON before write
                general_ideas = general_ideas
                f.write(general_ideas)
            
            with open("./prompt_library/wordlist_creation/2_suggests_filepath", 'r') as f:
                wordlistprompt = f.read()
            wordlistoutput = hn_assistant.run("Return plaintext list of URL only, no description : " + wordlistprompt + general_ideas, stream=False)
            #st.write("Generating Wordlist")
            #st.write(wordlistoutput)
            #Clean the wordlistoutput to JSON only
            wordlistoutput_clean = wordlistoutput
            # Store the wordlist output
            with open(f"{newpath}/wordlist_output.txt", "w") as f:
                f.write(wordlistoutput_clean)
        else:
            # Read the existing general ideas result
            with open(f"{newpath}/general_ideas_result.txt", "r") as f:
                general_ideas = f.read()
            #st.write("Generating Ideas")
            #st.write(general_ideas)
            
            # Read the existing wordlist output
            with open(f"{newpath}/wordlist_output.txt", "r") as f:
                wordlistoutput = f.read()
            #st.write("Generating Wordlist")
            #st.write(wordlistoutput)      
    

        st.write("Step 8 : Wordlist Creator")
        
        software_list = [folder for folder in os.listdir('./prompt_library') if os.path.isdir(os.path.join('./prompt_library', folder))]
        wappalyzer_software = list(wappalyzer_return.keys())
        
        matched_software = check_software_in_json(wappalyzer_return, software_list)

        #If matched found, then run the prompt for the software for seeder
        if matched_software:
            #st.write("Matched Software to start seeder:")
            #st.write(matched_software)
            for software in matched_software.keys():
                st.write(f"Running prompts for {software}...")
                software_prompts_path = os.path.join('./prompt_library', software.lower())
                st.write("Prompt Path: " + software_prompts_path)
                prompt_files = sorted([file for file in os.listdir(software_prompts_path) if os.path.isfile(os.path.join(software_prompts_path, file))])
                combined_result = summarized_result + "\n" + cot_result + "\n" + owasp_result
                for prompt_file in prompt_files:    
                    st.write(f"Executing prompt: {prompt_file}")
                    with open(os.path.join(software_prompts_path, prompt_file), 'r') as f:
                        software_prompt = f.read()
                    #Check if the output folder exists then skip the execution
                    if PRODUCTION:
                        software_tech_wordlist = hn_assistant.run(software_prompt + combined_result, stream=False)
                        st.write(software_tech_wordlist)
                        #Write the hn_assistant.run to the file
                        with open(f"{newpath}/software_prompt_{software}_{prompt_file}", "w") as f:
                            f.write(software_tech_wordlist)
                    else:
                        software_tech_wordlist = open(f"{newpath}/software_prompt_{software}_{prompt_file}", 'r').read()
                        st.write(software_tech_wordlist)



                    #if prompt_file == "1_find_wordpress_admin_path":
                    #    with open("./prompt_library/wordpress/1_find_wordpress_admin_path", 'r') as f:
                    #        wordpressprompt = f.read()
                    #    st.write(hn_assistant.run(wordpressprompt + combined_result, stream=False))
        
        else:
            st.write("No matched software found in Wappalyzer return.")
        
        
        
        st.write("Step 10 : Combined the generated wordlist and do the first execution")
        combined=wordlistoutput+general_ideas
        st.write(combined)


       
        def create_database(db_path):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS successful_urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS unsuccessful_urls (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    url TEXT UNIQUE
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS scan_summary (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    successful_count    ful_count INTEGER,
                    unsuccessful_count  INTEGER
                )
            ''')
            conn.commit()
            conn.close()

        def save_results_to_database(db_path, successful_urls, unsuccessful_urls):
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.executemany('''
                INSERT OR IGNORE INTO successful_urls (url) VALUES (?)
            ''', [(url,) for url in successful_urls])
            
            cursor.executemany('''
                INSERT OR IGNORE INTO unsuccessful_urls (url) VALUES (?)
            ''', [(url,) for url in unsuccessful_urls])
            
            cursor.execute('''
                INSERT INTO scan_summary (successful_count, unsuccessful_count) VALUES (?, ?)
            ''', (len(successful_urls), len(unsuccessful_urls)))
            
            conn.commit()
            conn.close()

        # Define the database path
        db_path = f"{newpath}/url_scan_results.db"

        # Create the database and tables if they do not exist
        create_database(db_path)

        # Step 11: Check if the URLs exist (using multithreading)
        st.write("Step 11: Check if the URLs exist (using multithreading)")

        # Using the combined variable, read the record one by one and combine with the domain name, try the URLs to see if they exist, save it as JSON format

        # Split the combined wordlist into individual URLs
        wordlist_urls = combined.split()
        #Save the wordlist to a file as step10_wordlist.txt
        #Check if the file exists, if yes, read the file when debugging is enabled
        if os.path.exists(f"{newpath}/step10_wordlist.txt"):

            with open(f"{newpath}/step10_wordlist.txt", "r") as f:
                wordlist_urls = f.read()
   

        else:
        # Initialize the results dictionary
            results = {"existing_urls": [], "non_existing_urls": []}

        # Check if the successful_urls.txt exists, if yes, read the file and append to the results
        if os.path.exists(f"{newpath}/successful_urls.txt"):
            with open(f"{newpath}/successful_urls.txt", "r") as f:
                results["existing_urls"] = f.read().split()

        # Check if the unsuccessful_urls.txt exists, if yes, read the file and append to the results
        if os.path.exists(f"{newpath}/unsuccessful_urls.txt"):
            with open(f"{newpath}/unsuccessful_urls.txt", "r") as f:
                results["non_existing_urls"] = f.read().split()

        # Check if the URL existence checking has already been completed
        if len(results["existing_urls"]) + len(results["non_existing_urls"]) == len(wordlist_urls):
            st.write("URL existence checking has already been completed.")
        else:
                    

                    # Function to check if a URL exists
            def check_url_existence(path):
                        
                        if path.startswith("http://") or path.startswith("https://"):
                            full_url = path
                        else:
                            full_url = f"http://{urlinfo.netloc}/{path.lstrip('/')}"
                        st.write(f"Checking {full_url}")
                        try:
                            response = requests.head(full_url, timeout=5)
                            if response.status_code == 200:
                                return full_url, True
                            else:
                                return full_url, False
                        except requests.RequestException:
                            return full_url, False

            # Create a progress bar
            progress_bar = st.progress(0)
            total_urls = len(wordlist_urls)

                # Use ThreadPoolExecutor to check URLs concurrently
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                    future_to_url = {executor.submit(check_url_existence, path): path for path in wordlist_urls}
                    for idx, future in enumerate(concurrent.futures.as_completed(future_to_url)):
                        url, exists = future.result()
                        if exists:
                            results["existing_urls"].append(url)
                        else:
                            results["non_existing_urls"].append(url)
                        # Update progress bar
                        progress_bar.progress((idx + 1) / total_urls)

                # Save the results to a file
            results_filename = f"{newpath}/url_existence_results.json"
            with open(results_filename, "w") as f:
                json.dump(results, f, indent=4)

            # Save the successful URLs to a separate file
            successful_urls_filename = f"{newpath}/successful_urls.txt"
            with open(successful_urls_filename, "w") as f:
                for url in results["existing_urls"]:
                        f.write(url + "\n")

                # Save the unsuccessful URLs to a separate file
            unsuccessful_urls_filename = f"{newpath}/unsuccessful_urls.txt"
            with open(unsuccessful_urls_filename, "w") as f:
                for url in results["non_existing_urls"]:
                    f.write(url + "\n")

            # Save the number of successful and unsuccessful URLs to a separate file
            summary_filename = f"{newpath}/url_check_summary.json"
            summary = {
                    "successful_count": len(results["existing_urls"]),
                    "unsuccessful_count": len(results["non_existing_urls"])
                }
            with open(summary_filename, "w") as f:
                    json.dump(summary, f, indent=4)

            # Save results to the database
            save_results_to_database(db_path, results["existing_urls"], results["non_existing_urls"])

            st.write("URL existence checking completed.")
            st.write(f"Results saved in {results_filename}")
            st.write(f"Successful URLs saved in {successful_urls_filename}")
            st.write(f"Unsuccessful URLs saved in {unsuccessful_urls_filename}")
            st.write(f"Summary saved in {summary_filename}")

        # List out the found URLs
        st.write("Existing URLs:")
        for url in results["existing_urls"]:
            st.write(url)

        st.write("Non-Existing URLs:")
        for url in results["non_existing_urls"]:
            st.write(url)
# Step 12: Read existing URLs from the database
    #db_path = f"{newpath}/url_scan_results.db"
    #successful_urls, unsuccessful_urls = read_existing_urls(db_path)
    #st.write("Existing URLs:", successful_urls)
    st.write("Step 12: Read existing URLs from the database")
    tried_count = 0
    successful_urls = []
    unsuccessful_urls = []
    while tried_count < 1000:
        
    #    st.write(f"Attempt {tried_count + 1}/{MAX_RANDOM_WORDLIST_TRIES}")
    #    st.write("successful_urls : " , successful_urls)
    #    st.write("matched_software.keys() " , matched_software.keys())
        #successful_urls, unsuccessful_urls = read_existing_urls(db_path)
        successful_urls = [url for url in successful_urls if url is not None]
        additional_paths = generate_more_paths(hn_assistant.run, "Herewith the workable URLs :"+ "\n".join(successful_urls)+ "\n".join(results["existing_urls"])+"Using the URL found on the website, give 50 more suggestions that would help in discover possible vulnerablities on the server, plaintext URL only, no number, no other information. Also refered to the summarized information of the target : "+combined_result+'NO EXAMPLE.COM, use the original domain name', matched_software.keys())
        #st.write("Additional paths generated:")
        #st.write("Output debug : "+additional_paths)
        #count the number of rows additional paths
        #Clean the array only web paths, remove all the components that not in URL format
        additional_paths = [path for path in additional_paths if path.startswith("http://") or path.startswith("https://")]
        numberofrow=len(additional_paths)
        st.write(len(additional_paths))
        tried_count=numberofrow+tried_count
        st.write("Total tried count : ", tried_count)

        # Check if the URLs exist
        for path in additional_paths:
            if path in successful_urls or path in unsuccessful_urls:
                continue

            full_url = f"http://{urlinfo.netloc}/{path.lstrip('/')}"
            full_url = clean_url(full_url)
            exists = check_url_existence(path)[1]

            if exists:
                successful_urls.append(full_url)
                #full url is a variable not a list, is it OK? 
                
                
                st.write(f"URL exists: {full_url}")
            else:
                unsuccessful_urls.append(full_url)
                
                st.write(f"URL does not exist: {full_url}")

    save_results_to_database(db_path, successful_urls, [])
    save_results_to_database(db_path, [], unsuccessful_urls)
    end_time = time.time()
    duration = end_time - start_time
    st.write(f"Total time taken to complete the program: {duration:.2f} seconds")
    #        #update_tried_number(db_path, full_url)
    #        #tried_count += 1

    #    tried_count += 1

    # Save results to the database after the loop
    #save_results_to_database(db_path, successful_urls, unsuccessful_urls)

    #st.write("URL existence checking completed after additional tries.")
    #st.write("Existing URLs:", successful_urls)
    #st.write("Non-Existing URLs:", unsuccessful_urls)

# Ensure the 'tried_count' column exists in the database
    



          
