
import json
import os
import base64

# Use the provided temporary directory
temp_dir = r'C:\Users\k.jonathan\.gemini\tmp\1ed76f67e25af6a0e4d481d6ea357d685a8bb49d5b01f84523609fad27189ce3'
notebook_path = os.path.join(temp_dir, 'temp_notebook_content.json')
output_path = os.path.join(temp_dir, 'parsed_notebook_content.txt')

def parse_notebook(input_path, output_path):
    try:
        with open(input_path, 'r', encoding='utf-16') as f:
            notebook_content = json.load(f)

        with open(output_path, 'w', encoding='utf-8') as out_f:
            for i, cell in enumerate(notebook_content['cells']):
                out_f.write(f"--- CELL {i+1}: {cell['cell_type'].upper()} ---\n\n")

                if 'source' in cell and cell['source']:
                    out_f.write("### SOURCE CODE/MARKDOWN ###\n")
                    out_f.write("".join(cell['source']))
                    out_f.write("\n\n")

                if 'outputs' in cell and cell['cell_type'] == 'code':
                    out_f.write("### OUTPUTS ###\n")
                    for output in cell['outputs']:
                        if output['output_type'] == 'stream':
                            out_f.write("--- Stream (stdout) ---\n")
                            out_f.write("".join(output.get('text', '')))
                            out_f.write("\n")
                        elif output['output_type'] == 'execute_result' or output['output_type'] == 'display_data':
                            if 'data' in output:
                                if 'text/plain' in output['data']:
                                    out_f.write("--- Text Output ---\n")
                                    out_f.write("".join(output['data']['text/plain']))
                                    out_f.write("\n")
                                if 'image/png' in output['data']:
                                    out_f.write("--- Image Output (e.g., a plot) ---\n")
                                    out_f.write("[An image was generated here. It is not displayed in this text format, but its presence is noted.]\n")
                        elif output['output_type'] == 'error':
                            out_f.write(f"--- Error ---\n")
                            out_f.write(f"ERROR: {output.get('ename', 'Unknown Error')}: {output.get('evalue', '')}\n")
                    out_f.write("\n")

    except FileNotFoundError:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(f"Error: Notebook file not found at {input_path}")
    except Exception as e:
        with open(output_path, 'w', encoding='utf-8') as out_f:
            out_f.write(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    parse_notebook(notebook_path, output_path)
