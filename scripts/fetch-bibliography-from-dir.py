import os
import yaml

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
                    metadata_list.append(front_matter)
    return metadata_list

if __name__ == "__main__":
    directory_path = 'posts/'  # Replace with your directory path containing Markdown files
    metadata = fetch_metadata_from_markdown_files(directory_path)
    for idx, data in enumerate(metadata, start=1):
        print(f"Metadata for file {idx}:")
        for key, value in data.items():
            print(f"{key}: {value}")
        print("-" * 30)