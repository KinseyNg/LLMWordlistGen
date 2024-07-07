# Import the required libraries
import streamlit as st
from scrapegraphai.graphs import SmartScraperGraph,SmartScraperMultiGraph
from Wappalyzer import Wappalyzer, WebPage
import pandas as pd
import streamlit as st
from phi.assistant import Assistant
from phi.tools.hackernews import HackerNews
from phi.llm.openai import OpenAIChat
from urllib.parse import urlparse
import os
import requests
from bs4 import BeautifulSoup
import difflib
from dotenv import load_dotenv
# Load the .env file
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")

PRODUCTION = True

def fetch_html(url):
    #st.write("Checking : " + url)
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
    #st.write(seq.ratio())
    return seq.ratio() > threshold

def filter_unique_urls(urls, base_domain, similarity_threshold=0.9):
    unique_urls = []
    html_structures = []
    st.write("Checking URLs and removing the URL with similar context")
    for url in urls:
        #st.write("Checking : "+url)
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
            #st.write(url)
            unique_urls.append(url)
            html_structures.append(html_structure)
    
    return unique_urls

def check_software_in_json(json_data, software_list):
        software_list_lower = [item.lower() for item in software_list]
        matched_software = {}

        for software, details in json_data.items():
            if software.lower() in software_list_lower:
                matched_software[software] = details
        
        return matched_software

def replace_prompt_reserved_text(prompt,list):
    #read the prompt , find any {} tag, such as {name},then replace it with the dict from list with same name
    for key, value in list.items():
        prompt = prompt.replace("{{"+key+"}}", value)
    return prompt

def read_prompt(mode, type_text, item, source, datalist=[]):
    def run_with_error_handling(content):
        #st.write("run with error handing")
        try:
            return hn_assistant.run(content, stream=False)
        except Exception as e:
            print(f"Error occurred during hn_assistant.run: {e}")
            if len(content) > 1:  # Ensure there's something to split
                # Split the content into two halves
                mid = len(content) // 2
                part1 = content[:mid]
                part2 = content[mid:]
                # Run each part separately with further error handling
                result1 = run_with_error_handling(part1)
                result2 = run_with_error_handling(part2)
                # Combine and return results
                combined_result = result1 + result2
                return combined_result
            else:
                # If content cannot be split further, return an error message
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
        response = run_with_error_handling(combined_content)

    elif mode == "exe_all":
        prompt_library_path = os.path.join('./prompt_library', type_text.lower())
        st.write(prompt_library_path)
        prompt_list = []
        for root, dirs, files in os.walk(prompt_library_path):
            for file in files:
                with open(os.path.join(root, file), "r") as f:
                    prompt_list.append(f.read())
        if datalist:
            combined_prompts = replace_prompt_reserved_text(prompt_list, datalist)
        else:
            combined_prompts=prompt_list
        st.write(combined_prompts)
        response = run_with_error_handling(combined_prompts)

    return response
#for root, dirs, files in os.walk(prompt_library_path):
#            for file in files:
#                if file.endswith(".txt"):
#                    with open(os.path.join(root, file), "r") as f:
#                        prompt_list.append(f.read())


# Get OpenAI API key from user
openai_access_token = api_key
openai_api_key = openai_access_token
def generate_response(input_text):
    llm = OpenAI(temperature=0.7, openai_api_key=openai_access_token)
    return llm(input_text)

# Set up the Streamlit app
st.title("LLM-Based Wordlist Generation Framework")
st.caption("by Kinsey v0.1")

if openai_access_token:
    model = st.radio(
        "Select the model",
        ["gpt-3.5-turbo-0125", "gpt-4","gpt-4-1106-preview"],
        index=0,
    )
    hn_assistant = Assistant(
        name="LLM Wordlist Creator",
        #team=[story_researcher, user_researcher],
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
    
    #Classifcation to different framework -> Become different prompts -> faster progress 
    #More table more results : 10 Websites 
    # 準確/ coverage 
    # 漏洞 / testing 
    
    #Summmary 
    #1. Technology used
    #2. 



    # Get the URL of the website to scrape
    #Coverage ? 
    url = st.text_input("Enter the URL of the website you want to scrape")
    # Get the user prompt
    scraper_step1 = "You are a web elements extractor. Your task is to help extract HTML elements of all hyperlinks with complete HTML components. \
    List only HTTP links and return in a structured format. Also, 	Extract the JavaScript code and library URL used in the given HTML code. 	For each found HTML code component, remove all the code that is not related to the main business of the website, such as:\
    Advertisement code \
    CSS codes \
    Image related code or other items with similar direction \ Also,	Extract the Javascript AJAX call if any \
    	Combine and construct all the results into JSON format. Example : "
        
    scraper_step1=scraper_step1+ """
    {
        "Domain": "www.example.com",
        "Server": "Apache 2.3.4",
        "URL": "[Array of Multiple URLs]",
        "HTMLCodes": "[Array of scraped HTML components]",
        "JavaScript": "[Array of scraped JavaScript components]"
    }
    """

    urlinfo=urlparse(url)
    base_domain = urlinfo.netloc
    #ParseResult(scheme='http', netloc='somesite.com', path='/page.php', params='', query='id=sas231', fragment='')
    # Create a SmartScraperGraph object
    smart_scraper_graph = SmartScraperGraph(
        prompt=scraper_step1,
        source=url,
        config=graph_config
    )
    # Scrape the website
    if st.button("Start"):
        
        
        st.write("Step 1 : Identify the web framework and technologies with Wappalyzer")
        ##https://github.com/chorsley/python-Wappalyzer
        webpage = WebPage.new_from_url(url)
        wappalyzer = Wappalyzer.latest()
        wappalyzer_return=wappalyzer.analyze_with_categories(webpage)
        df = pd.DataFrame(wappalyzer_return)
        newpath = './output/'+urlinfo.netloc 
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        store_filename=newpath+'/server_info'
        f = open(store_filename, "w")
        f.write(str(wappalyzer_return))
        f.close()

        st.dataframe(df, use_container_width=True)
        st.write("Step 2 : Scrapping the URLs from the website")
        
        result = smart_scraper_graph.run()
        
        #Loop the array to read the url one by one, crawl the page and read the url again, add up all to listtourl
        #listofurl=[]
        #for url in result["URL"]:
        #    smart_scraper_graph = SmartScraperGraph(
        #        prompt=scraper_step1,
        #        source=url,
        #        config=graph_config
        #        )
        #    result = smart_scraper_graph.run()
            

        st.write(result["URL"])
        


        ##Can compare depth setting for the effectiveness of generating wordlist?

        st.write("Step 3 : Find the unique URL from step 2 output")
        st.write("Filtering unique URLs from the list of URLs extracted from the website")
        filtered_urls = filter_unique_urls(result["URL"],base_domain,0.5) 
        st.write("List of unique URLs")
        st.write(filtered_urls)

        st.write("Step 4 : Crawl all the discovered URLs and save it HTML contents")
        scrapped_html_path = f'{newpath}/scrapped_html'
        if not os.path.exists(scrapped_html_path):
            os.makedirs(scrapped_html_path)

        for unique_url in filtered_urls:
            html_content = fetch_html(unique_url)
            if html_content:
                url_path = urlparse(unique_url).path.strip("/").replace("/", "_")
                if url_path=="":
                    url_path = "index"
                html_filename = f'{scrapped_html_path}/{url_path}'
                #st.write(html_filename)
                with open(html_filename, "w") as html_file:
                    html_file.write(html_content)
        st.write("Saved all in "+scrapped_html_path)
        st.write("Step 5 : Feature Extractor, Summarizer, and Classifier")
        summarized_results_html_path = f'{newpath}/summarized_results'
        #Read all saved html
        summarized_result = ""
        processed_files = 0
        total_files = sum(len(files) for _, _, files in os.walk(scrapped_html_path))
        my_bar = st.progress(0, text="Analyzing progress")
        for root, dirs, files in os.walk(scrapped_html_path):
            for file in files:
                with open(os.path.join(root, file), "r") as f:
                    st.write("Analyzing : "+file)
                    if PRODUCTION:
                        summarized_result += read_prompt("exe_one","basic_web","1_summarize_theme",f.read(),"")  
                        with open(summarized_results_html_path, "w") as f:
                            f.write(summarized_result)
                            st.write("Saved LLM summarizing results in "+summarized_results_html_path)
                    else:
                        summarized_result = open(summarized_results_html_path, 'r').read() 
                    # Update progress bar
                    processed_files += 1
                    progress = processed_files / total_files
                    my_bar.progress(progress, text=f"Analyzing progress: {int(progress * 100)}%")

        st.write("Summarized results")
        st.write(summarized_result)

        st.write("Step 6 : CoT for guessing the development process")
        cot_path = f'{newpath}/cot_results'
        cot_result = read_prompt("exe_one","basic_web","2_cot_misconfig","Framework used by wappalyzer : "+str(wappalyzer_return)+"Website Description : "+summarized_result,"")  
        with open(cot_path, "w") as f:
                            f.write(cot_path)
                            st.write("Saved CoT results in "+cot_path)
        #st.write(cot_result)

        st.write("Step 7 : Run OWASP check")
        owasp_path = f'{newpath}/owasp_results'
        owasp_result = read_prompt("exe_one","basic_web","3_OWASP","Framework used by wappalyzer : "+str(wappalyzer_return)+"Website Technoglogies information : "+cot_result,"")  
        with open(owasp_path, "w") as f:
                            f.write(summarized_result)
                            st.write("Saved OWASP check results in "+owasp_path)
        #st.write(owasp_result)


        st.write("Step 8 : Wordlist Creator")
        #Consider CMS or non CMS

        # Get the available software list by reading subfolder names
        software_list = [folder for folder in os.listdir('./prompt_library') if os.path.isdir(os.path.join('./prompt_library', folder))]
        wappalyzer_software = list(wappalyzer_return.keys())
        
        matched_software = check_software_in_json(wappalyzer_return, software_list)

        if matched_software:
            st.write("Matched Software:")
            st.write(matched_software)
            for software in matched_software.keys():
                st.write(f"Running prompts for {software}...")
                software_prompts_path = os.path.join('./prompt_library', software.lower())
                prompt_files = sorted([file for file in os.listdir(software_prompts_path) if os.path.isfile(os.path.join(software_prompts_path, file))])
                combined_result = summarized_result + "\n" + cot_result + "\n" + owasp_result
                for prompt_file in prompt_files:    
                    st.write(f"Executing prompt: {prompt_file}")
                    if prompt_file == "1_find_wordpress_admin_path":
                        with open("./prompt_library/wordpress/1_find_wordpress_admin_path", 'r') as f:
                            wordpressprompt= f.read()
                        st.write(hn_assistant.run(wordpressprompt+combined_result, stream=False))
                    #else:
                       

                    
                #(TEMP) Additional code for handling PHP and WordPress
                #if "php" in matched_software.keys() or "wordpress" in matched_software.keys():
                #    st.write("Matched!")
                #    additional_software_prompts_path = os.path.join('./prompt_library', 'php')
                #    combined_result = summarized_result + "\n" + cot_result + "\n" + owasp_result
                #    if "wordpress" in matched_software.keys():
                #        additional_software_prompts_path = os.path.join('./prompt_library', 'wordpress')
                        
                    #additional_prompt_files = sorted([file for file in os.listdir(additional_software_prompts_path) if os.path.isfile(os.path.join(additional_software_prompts_path, file))], key=lambda x: int(x.split('_')[0]))

                    #for additional_prompt_file in additional_prompt_files:
                    #    st.write(f"Executing additional prompt: {additional_prompt_file}")
                    #    st.write(hn_assistant.run(f.read(additional_prompt_file), stream=False))
                    #    #combined_result = read_prompt("exe_one", additional_software_prompts_path.split('/')[-1], additional_prompt_file, combined_result, matched_software)
                    #    st.write(combined_result)
                #for root, dirs, files in os.walk(software_prompts_path):
                #    for file in files:
                #        with open(os.path.join(root, file), "r") as f:
                #            prompt_content = f.read()
                #            response = read_prompt("exe_all", software, file, summarized_result, matched_software)
                #            st.write(f"Prompt: {file}")
                #            st.write(response)
        else:
            st.write("No matched software found in Wappalyzer return.")
            with open("./prompt_library/wordlist_creation/1_provide_ideas", 'r') as f:
                        wordpressprompt= f.read()
                        combined_result = summarized_result + "\n" + cot_result + "\n" + owasp_result
                        general_ideas= hn_assistant.run(wordpressprompt+combined_result, stream=False)
                        st.write("Generating Ideas")
                        st.write("No matched software found in Wappalyzer return.")
                        with open("./prompt_library/wordlist_creation/2_suggests_filepath", 'r') as f:
                            wordlistprompt= f.read()

                        wordlistoutput= hn_assistant.run(wordlistprompt+general_ideas, stream=False)
                        st.write("Generating Wordlist")
                        st.write(wordlistoutput)





        