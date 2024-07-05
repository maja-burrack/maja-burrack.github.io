import os
import yaml
import json
from datetime import datetime

AUTHORS_TO_REMOVE = [
    'Geneen Green'
]

directory_path = '/Users/majalarsen/Library/Mobile Documents/iCloud~md~obsidian/Documents/Obsidian Vault/sources/bibliography'

def fetch_metadata_from_markdown_files(directory):
    metadata_list = []
    for filename in os.listdir(directory):
        if filename.endswith('.md'):
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                if lines[0].strip() == '---':  # Assuming YAML front matter is present
                    front_matter_lines = []
                    i = 1
                    while lines[i].strip() != '---':
                        front_matter_lines.append(lines[i])
                        i += 1
                        
                    front_matter = yaml.safe_load('\n'.join(front_matter_lines))
                    # Filter for entries with both completion_date and title
                    # if 'completion_date' in front_matter and front_matter['completion_date'] and 'title' in front_matter:
                    metadata_list.append(front_matter)
    return metadata_list

def filter_metadata(metadata, authors_to_remove=AUTHORS_TO_REMOVE):
    # remove unwanted authors
    metadata_filtered = [item for item in metadata if item['author'] not in authors_to_remove]
    # remove unfinished books
    metadata_filtered = [item for item in metadata_filtered if item['completed'] is not None]
    # remove any accidental files without a title
    metadata_filtered = [item for item in metadata_filtered if item['title'] is not None]
    return metadata_filtered


if __name__ == "__main__":
    directory_path = input('path to md files:')  # Replace with your directory path containing Markdown files
    metadata = fetch_metadata_from_markdown_files(directory_path)
    metadata_filtered = filter_metadata(metadata)
    
    # Write metadata to JSON file
    json_file_path = 'metadata.json'  # Path to save the JSON file
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(metadata_filtered, json_file, ensure_ascii=False, indent=4, default=str)

    print(f"Metadata written to {json_file_path}")
    