import wikipedia
import json
import re
import time
import logging
from multiprocessing import Pool, Manager, cpu_count
from functools import partial
import sys
import os
import random
from bs4 import BeautifulSoup
from requests.exceptions import HTTPError
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse, urlunparse
import hashlib

# Suppress specific warnings from BeautifulSoup
import warnings
from bs4 import GuessedAtParserWarning
warnings.filterwarnings("ignore", category=GuessedAtParserWarning)

# ----------------- Enhanced Logging Configuration -----------------
# Create a custom logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)  # Set the minimum logging level

# Define the log format
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Rotating file handler for logging to a file
file_handler = RotatingFileHandler('wikipedia_scraper.log', maxBytes=10**7, backupCount=5)  # 10MB per file, up to 5 backups
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Stream handler for logging to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)
# --------------------------------------------------------------------

# Configuration Parameters
CONFIG = {
    "topics": {
        "Health": [
            "Health and Medicine",
            "Healthcare",
            "Mental Health",
            "Public Health",
            "Medical Health Practices",
            "Nutrition",
            "Diseases",
            "Medical Treatments",
            "Healthcare Policy",
            "Pharmaceuticals",
            "Preventive Medicine",
            "Health Technology",
            "Epidemiology",
            "Healthcare Management",
            "Reproductive Health"
        ],
        "Environment": [
            "Climate Change",
            "Environmental Protection",
            "Sustainability",
            "Conservation",
            "Environmental Science",
            "Ecology",
            "Pollution",
            "Renewable Energy",
            "Biodiversity",
            "Natural Resources",
            "Waste Management",
            "Environmental Policy",
            "Green Technology",
            "Habitat Restoration",
            "Marine Conservation"
        ],
        "Technology": [
            "Technology",
            "Technological Advancements",
            "Innovative Technologies",
            "Technology Applications",
            "Artificial Intelligence",
            "Machine Learning",
            "Information Technology",
            "Cybersecurity",
            "Robotics",
            "Nanotechnology",
            "Blockchain Technology",
            "Internet of Things (IoT)",
            "Virtual Reality (VR)",
            "Augmented Reality (AR)",
            "Quantum Computing"
        ],
        "Economy": [
            "Economics",
            "Global Economy",
            "Economic Policy",
            "Financial Markets",
            "Economic Development",
            "International Trade",
            "Fiscal Policy",
            "Monetary Policy",
            "Economic Theory",
            "Labor Economics",
            "Behavioral Economics",
            "Development Economics",
            "Environmental Economics",
            "Health Economics",
            "Public Economics"
        ],
        "Entertainment": [
            "Film Industry",
            "Television",
            "Music",
            "Video Games",
            "Theater",
            "Dance",
            "Literature",
            "Art",
            "Celebrity Culture",
            "Streaming Services",
            "Animation",
            "Stand-up Comedy",
            "Podcasting",
            "Digital Media",
            "Live Events"
        ],
        "Sports": [
            "Sport",
            "Sports Disciplines",
            "Athletics",
            "Competitive Athletics",
            "Olympics",
            "Team Sports",
            "Individual Sports",
            "Sports Management",
            "Sports Medicine",
            "E-Sports",
            "Extreme Sports",
            "Sports Psychology",
            "Sports Training",
            "Sports Analytics",
            "Sports Marketing"
        ],
        "Politics and Government": [
            "Politics and Government",
            "Political Governance",
            "Government Politics",
            "Public Administration",
            "Democracy",
            "Political Theory",
            "International Relations",
            "Public Policy",
            "Legislation",
            "Political Parties",
            "Public Opinion",
            "Political Campaigns",
            "Government Agencies",
            "Comparative Politics",
            "Political Economy"
        ],
        "Education": [
            "Education",
            "Educational Systems",
            "Academic Topics",
            "Educational Reforms",
            "Online Learning",
            "Higher Education",
            "Primary Education",
            "Secondary Education",
            "Educational Technology",
            "Special Education",
            "Early Childhood Education",
            "Adult Education",
            "Vocational Training",
            "Educational Policy",
            "STEM Education"
        ],
        "Travel": [
            "Tourism",
            "Adventure Travel",
            "Cultural Tourism",
            "Sustainable Tourism",
            "Travel Destinations",
            "Travel Guides",
            "Hospitality Industry",
            "Travel Safety",
            "Ecotourism",
            "Travel Photography",
            "Backpacking",
            "Luxury Travel",
            "Business Travel",
            "Solo Travel",
            "Travel Blogging"
        ],
        "Food": [
            "Culinary Arts",
            "World Cuisines",
            "Nutrition",
            "Food Safety",
            "Gastronomy",
            "Food Culture",
            "Recipes",
            "Food Science",
            "Dietary Practices",
            "Food Industry",
            "Baking",
            "Vegetarian and Vegan Foods",
            "Street Food",
            "Food Sustainability",
            "Beverages"
        ]
    },
    "min_documents_per_category": 5000,
    "max_documents_per_category": 5500,
    "max_summary_length": 500,
    "min_summary_length": 200,
    "short_summary_threshold_percentage": 0.05,
    "retry_pause_seconds": 5,
    "max_retries": 10,
    "output_files": {
        "valid": "wikipedia_scraped_docs.json",
        "invalid": "wikipedia_invalid_docs.json"
    },
    "processes": max(1, cpu_count() - 1),  # Leave one CPU free
    "batch_size": 2,  # Number of topics to process before a pause
    "pause_duration_minutes": [5, 10]  # Range for pause duration in minutes
}

def clean_title(title):
    return re.sub(r'[^A-Za-z0-9 ]+', '', title).strip()

def clean_description(summary):
    return re.sub(r'[^A-Za-z0-9 ]+', ' ', summary).strip()

def normalize_url(url):
    """Normalize the URL to ensure consistency."""
    parsed = urlparse(url)
    # Remove query parameters and fragments
    parsed = parsed._replace(query="", fragment="")
    # Convert scheme and netloc to lowercase
    parsed = parsed._replace(scheme=parsed.scheme.lower(), netloc=parsed.netloc.lower())
    return urlunparse(parsed)

def hash_url(url):
    """Hash the URL using MD5 to reduce memory usage."""
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def fetch_wikipedia_articles(category, keywords, config, scraped_docs, invalid_pages, checked_urls_dict):
    minimum = config["min_documents_per_category"]
    maximum = config["max_documents_per_category"]
    attempts = len(keywords)-1
    pause = config["retry_pause_seconds"]
    max_short_summaries = int(config["short_summary_threshold_percentage"] * minimum)
    
    brief_summary_counter = 0
    total_trials = attempts + 1  # Initial attempt plus retries
    doc_count = 0

    for trial in range(total_trials):
        if trial < len(keywords):
            current_keyword = keywords[trial]
        else:
            current_keyword = keywords[-1]
            logger.warning(f"No additional unique subtopics for '{category}'. Reusing the last subtopic.")
        
        logger.info(f"Scrape Attempt {trial + 1} for '{category}' using subtopic '{current_keyword}'")
        valid_count = 0
        invalid_count = 0
        try:
            search_results = wikipedia.search(current_keyword, results=maximum * 2)
        except Exception as e:
            logger.error(f"Search failed for '{category}' with subtopic '{current_keyword}': {e}")
            invalid_pages.append(f"Search failed with subtopic '{current_keyword}': {e}")
            invalid_count += 1
            continue
        
        for title in search_results:
            if doc_count >= maximum:
                logger.info(f"Reached maximum document limit ({maximum}) for category '{category}'")
                break
            try:
                page = wikipedia.page(title, auto_suggest=False)
                cleaned_title = clean_title(page.title)
                
                if not cleaned_title:
                    logger.debug(f"Skipping page with invalid title: {page.title}")
                    invalid_pages.append(page.title)
                    invalid_count += 1
                    continue
                
                summary_text = clean_description(page.summary[:config["max_summary_length"]])
                
                if len(summary_text) < config["min_summary_length"]:
                    if brief_summary_counter >= max_short_summaries:
                        logger.debug(f"Skipping '{page.title}' due to short summary (Length: {len(summary_text)})")
                        invalid_pages.append(page.title)
                        invalid_count += 1
                        continue
                    brief_summary_counter += 1
                
                normalized_url = normalize_url(page.url)
                url_hash = hash_url(normalized_url)
                
                if url_hash in checked_urls_dict:
                    logger.debug(f"Duplicate URL found. Skipping '{title}' with URL: {normalized_url}")
                    invalid_pages.append(title)
                    invalid_count += 1
                    continue
                else:
                    checked_urls_dict[url_hash] = True  # Mark URL as seen
                
                document_data = {
                    "revision_id": page.pageid,
                    "title": cleaned_title,
                    "summary": summary_text,
                    "url": normalized_url,
                    "topic": category
                }
                
                scraped_docs.append(document_data)
                valid_count += 1
                doc_count += 1
                logger.debug(f"Scraped document: '{cleaned_title}' (ID: {page.pageid})")
            
            except wikipedia.exceptions.DisambiguationError as e:
                logger.debug(f"Disambiguation page encountered. Skipping '{title}'")
                invalid_pages.append(title)
                invalid_count += 1
            except wikipedia.exceptions.PageError:
                logger.debug(f"Page not found: '{title}'")
                invalid_pages.append(title)
                invalid_count += 1
            except HTTPError as http_err:
                logger.error(f"HTTP error occurred for page '{title}': {http_err}")
                invalid_pages.append(title)
                invalid_count += 1
            except Exception as e:
                logger.error(f"Error processing page '{title}': {e}")
                invalid_pages.append(title)
                invalid_count += 1
        
        logger.info(f"After trial {trial + 1} for '{category}': {valid_count} valid pages scraped, {invalid_count} invalid pages encountered.")
        
        if doc_count >= minimum:
            logger.info(f"Target of {doc_count} documents scraped for '{category}'.")
            break
        else:
            logger.info(f"Scraped {doc_count} documents for '{category}'. Retrying after {pause} seconds...")
            time.sleep(pause)
    
    if doc_count < minimum:
        logger.warning(f"Only {doc_count} documents collected for '{category}' after {total_trials} attempts.")

def run_scraping():
    keyword_mapping = CONFIG["topics"]
    manager = Manager()
    collected_docs = manager.list()
    total_fails = manager.dict()
    checked_urls_dict = manager.dict()  # Shared dictionary to track scraped URLs
    
    # Initialize total_fails with manager.list() for each category
    for category in keyword_mapping.keys():
        total_fails[category] = manager.list()

    # Prepare a list of arguments for starmap
    args_list = []
    for category, keywords in keyword_mapping.items():
        args_list.append((category, keywords, CONFIG, collected_docs, total_fails[category], checked_urls_dict))

    # Convert topics to a list of tuples
    topics_list = list(keyword_mapping.items())
    total_topics = len(topics_list)
    batch_size = CONFIG["batch_size"]
    pause_min, pause_max = CONFIG["pause_duration_minutes"]
    logger.info(f"Total number of categories to process: {total_topics}")
    logger.info(f"Processing in batches of {batch_size} categories with pauses of {pause_min}-{pause_max} minutes between batches.")

    # Process topics in batches
    for batch_start in range(0, total_topics, batch_size):
        batch_end = min(batch_start + batch_size, total_topics)
        current_batch = args_list[batch_start:batch_end]
        batch_num = (batch_start // batch_size) + 1
        logger.info(f"Processing batch {batch_num}: Topics {batch_start + 1} to {batch_end}")

        with Pool(processes=CONFIG["processes"]) as pool:
            pool.starmap(
                fetch_wikipedia_articles,
                current_batch
            )
        
        if batch_end < total_topics:
            # Randomize pause duration between min and max minutes
            pause_duration = random.randint(pause_min, pause_max)
            logger.info(f"Batch {batch_num} completed. Pausing for {pause_duration} minutes to respect Wikipedia servers.")
            time.sleep(pause_duration * 60)  # Convert minutes to seconds

    # Convert Manager lists/dicts to regular lists/dicts
    final_docs = list(collected_docs)
    final_fails = {category: list(fails) for category, fails in total_fails.items()}

    # Remove duplicates based on revision_id
    unique_docs = {}
    for doc in final_docs:
        unique_docs[doc['revision_id']] = doc
    final_docx = list(unique_docs.values())

    # Save valid documents
    try:
        with open("dgusain_wikipedia_scraped_docs.json", "w", encoding='utf-8') as outfile:
            json.dump(final_docx, outfile, indent=4, ensure_ascii=False)
        with open(CONFIG["output_files"]["valid"], "w", encoding='utf-8') as outfile2:
            json.dump(final_docx, outfile2, indent=4, ensure_ascii=False)
        logger.info(f"Total unique documents scraped: {len(final_docx)}. Saved to '{CONFIG['output_files']['valid']}' | Added to 'dgusain_wikipedia_scraped_docs.json'")
    except Exception as e:
        logger.error(f"Failed to save valid documents: {e}")

    # Save invalid documents
    try:
        with open("dgusain_invalid_docs.json", "w", encoding='utf-8') as fail_file:
            json.dump(final_fails, fail_file, indent=4, ensure_ascii=False)
        with open(CONFIG["output_files"]["invalid"], "w", encoding='utf-8') as fail_file:
            json.dump(final_fails, fail_file, indent=4, ensure_ascii=False)
        logger.info(f"Invalid documents saved to '{CONFIG['output_files']['invalid']}' | Invalid documents have been saved to 'dgusain_invalid_docs.json'.")
    except TypeError as te:
        logger.error(f"TypeError during JSON serialization of invalid documents: {te}")
    except Exception as e:
        logger.error(f"Failed to save invalid documents: {e}")

    # Final Counts
    total_valid = len(final_docx)
    total_invalid = sum(len(fails) for fails in final_fails.values())
    logger.info(f"Final Counts: {total_valid} valid pages scraped, {total_invalid} invalid pages encountered.")
    logger.info("Wikipedia scraper has completed its execution.")

if __name__ == "__main__":
    try:
        start_time = time.time()
        logger.info("Wikipedia scraper started.")
        run_scraping()
        end_time = time.time()
        elapsed = end_time - start_time
        elapsed_minutes = elapsed / 60
        logger.info(f"Wikipedia scraper finished in {elapsed_minutes:.2f} minutes.")
    except Exception as e:
        logger.critical(f"Scraper encountered a critical error: {e}", exc_info=True)
        sys.exit(1)
