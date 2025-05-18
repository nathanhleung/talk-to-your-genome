import os
import subprocess
from functools import lru_cache
from typing import List, Dict, Any, Optional
import json  # For parsing tool input if necessary, and formatting tool output
import time
from dotenv import load_dotenv

load_dotenv()

import anthropic
import uvicorn
from fastapi import FastAPI, HTTPException, status as http_status
import vcfpy
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- New pharmcat_diplotypes function ---
@lru_cache
def pharmcat_diplotypes(genes: List[str]) -> Dict:
    """
    1. Mounts the directory of `vcf_path` into /data in the PharmCAT container.
    2. Runs the full pipeline with the -reporterCallsOnlyTsv flag.
    3. Finds the <basename>.report.tsv and parses out only the requested genes.
    """
    vcf_path = os.environ.get("VCF_FILE_PATH")
    if not vcf_path:
        return {
            "error": "VCF_FILE_PATH environment variable not set.",
            "diplotypes": {},
            "docker_command": ""
        }
    if not os.path.exists(vcf_path):
        return {
            "error": f"VCF file not found at path: {vcf_path}",
            "diplotypes": {},
            "docker_command": ""
        }

    workdir = os.path.dirname(vcf_path)
    base = os.path.basename(vcf_path)
    stem = os.path.splitext(base)[0]

    cmd = [
        "sudo" "docker", "run", "--rm",
        "-v", f"{workdir}:/data",
        "pgkb/pharmcat",
        "pharmcat_pipeline",
        "-reporterCallsOnlyTsv",
        f"/data/{base}"
    ]
    try:
        process_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        # print(f"PharmCAT stdout: {process_result.stdout}") # For debugging
        # print(f"PharmCAT stderr: {process_result.stderr}") # For debugging
    except subprocess.CalledProcessError as e:
        return {
            "error": f"PharmCAT Docker execution failed: {e.stderr}",
            "diplotypes": {},
            "docker_command": " ".join(cmd)
        }
    except FileNotFoundError:  # Docker command not found
        return {
            "error": "Docker command not found. Please ensure Docker is installed and in your PATH.",
            "diplotypes": {},
            "docker_command": " ".join(cmd)
        }

    tsv_path = os.path.join(workdir, f"{stem}.report.tsv")
    if not os.path.exists(tsv_path):
        return {
            "error": f"PharmCAT report TSV file not found: {tsv_path}. PharmCAT may not have generated the expected output. Docker logs (if any during run): {process_result.stderr}",
            "diplotypes": {},
            "docker_command": " ".join(cmd)
        }

    diplotypes = {}
    try:
        with open(tsv_path) as f:
            # Skip initial lines that might not be the header, robustly find header
            header_line = ""
            # Max lines to search for header (to prevent reading huge files if something is wrong)
            for _ in range(5):
                line = f.readline()
                if not line: break  # End of file
                if "Gene" in line and "Source Diplotype" in line:
                    header_line = line
                    break

            if not header_line:
                return {
                    "error": f"PharmCAT report TSV file ({tsv_path}) is missing the expected header ('Gene', 'Source Diplotype'). Content might be unexpected.",
                    "diplotypes": {},
                    "docker_command": " ".join(cmd)
                }

            header = header_line.strip().split("\t")

            try:
                gene_idx = header.index("Gene")
                dip_idx = header.index("Source Diplotype")
            except ValueError:
                return {
                    "error": f"PharmCAT report TSV file ({tsv_path}) is missing 'Gene' or 'Source Diplotype' column in the identified header.",
                    "diplotypes": {},
                    "docker_command": " ".join(cmd)
                }

            for line in f:
                cols = line.strip().split("\t")
                if len(cols) > max(gene_idx, dip_idx):
                    if cols[gene_idx] in genes:
                        diplotypes[cols[gene_idx]] = cols[dip_idx]
                else:
                    # Optional: log malformed lines if necessary
                    # print(f"Skipping malformed/short line in TSV: {line.strip()}")
                    pass
    except Exception as e:
        return {
            "error": f"Error parsing PharmCAT TSV report '{tsv_path}': {str(e)}",
            "diplotypes": {},
            "docker_command": " ".join(cmd)
        }

    return {
        "diplotypes": diplotypes
    }


# --- Pydantic Models ---
class SNPediaRequest(BaseModel):
    question: str
    snps: Optional[List[str]] = None
    token: str


class ChatResponse(BaseModel):
    text: str
    sources: List[str]


class BasePairsResponse(BaseModel):
    response: str


class SNPLocation(BaseModel):
    chromosome: int
    position: int


# --- User's original citation extraction function (renamed for clarity) ---
def extract_text_with_citations_from_sdk_blocks(api_response_content_blocks: List[Any]):
    user_texts = []
    citations_list = []  # Renamed to avoid conflict with 'citations' module

    for block_sdk_obj in api_response_content_blocks:
        block_dict = {}
        if hasattr(block_sdk_obj, 'type'):  # It's an SDK object
            if block_sdk_obj.type == "text":
                text_content = getattr(block_sdk_obj, 'text', "").strip()
                if text_content:
                    user_texts.append(text_content)
            # Try to convert to dict for citation extraction as per original logic
            # This relies on the SDK object being dict-convertible in a specific way
            try:
                block_dict = dict(block_sdk_obj)
            except Exception:  # Broad exception if dict() fails
                block_dict = {}  # Ensure block_dict is a dict for .get() calls
                # If dict conversion fails, direct attribute access for citations might be needed
                # e.g. if hasattr(block_sdk_obj, "citations_list_attr"): ...
                # For now, we rely on the original code's dict conversion assumption.
        elif isinstance(block_sdk_obj, dict):  # If it's already a dict
            block_dict = block_sdk_obj
            if block_dict.get("type") == "text":
                text_content = block_dict.get("text", "").strip()
                if text_content and text_content not in user_texts:  # Avoid duplicates if already added
                    user_texts.append(text_content)

        # Citation extraction from the (potentially converted) dictionary
        if block_dict.get("citations"):
            for cite_obj in block_dict.get("citations", []):
                cite_dict = {}
                try:
                    cite_dict = dict(cite_obj)
                except Exception:
                    # If cite_obj is not dict-convertible but has attributes
                    if hasattr(cite_obj, 'url') and hasattr(cite_obj, 'title'):
                        cite_dict = {'url': getattr(cite_obj, 'url'), 'title': getattr(cite_obj, 'title')}
                    else:  # Give up if no known structure
                        continue

                url = cite_dict.get("url")
                title = cite_dict.get("title")
                if url and title:
                    citations_list.append(f"{title}: {url}")

    full_text = "\n\n".join(user_texts)
    return full_text, list(set(citations_list))  # Return unique citations


# --- FastAPI Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def search_snpedia(request: SNPediaRequest):
    if request.token != 'texakloma':
        raise HTTPException(status_code=http_status.HTTP_401_UNAUTHORIZED, detail=f"Unauthorized")

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    SYSTEM_PROMPT = """
    You are a helpful assistant that provides information about SNPs and their effects on health.
    You will be given a list of SNPs the user has. They will ask questions and you should provide personalized health advice based on the SNPs provided.
    You should also provide sources where the user can get more information if needed. Omit any SNPs that are not relevant to the specific user question.
    Use simple language with direct answers.
    Responses should be a maximum of one paragraph.

    You have access to two tools:
    1. `web_search`: To search SNPedia.com for information about SNPs. Use this to answer questions about specific SNPs if detailed information from SNPedia is required.
    2. `pharmcat_diplotypes`: To run the PharmCAT Docker pipeline on the user's VCF data (implicitly available to the tool) and return a mapping of gene to star-allele diplotype for requested genes. Use this tool when the user's question pertains to pharmacogenomics, drug metabolism (e.g. for genes like CYP2C19), or requires diplotype information for specific genes mentioned or implied in the question. Only specify genes relevant to the user's question in the tool input.
    """
    messages = []
    if request.snps:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Here are my SNPs: {', '.join(request.snps)}."}
                ],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"Here is my question: {request.question}"}
            ],
        },
    )


    anthropic_tools_definition = [
        {
            "name": "web_search",
            "type": "web_search_20250305",  # Special type for Anthropic's integrated web search
            "allowed_domains": ["snpedia.com"],
        },
        {
            "name": "pharmcat_diplotypes",
            "description": "Run the PharmCAT Docker pipeline on the user's VCF and return a mapping of gene→star‑allele diplotype. The VCF file is pre-configured on the server.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of genes to process (e.g., GBA, LRRK2, CYP2C19). Only include genes relevant to the user's question."
                    }
                },
                "required": ["genes"]
            }
        }
    ]

    MAX_TOOL_ITERATIONS = 5  # Prevent infinite loops if model keeps calling tools
    current_iteration = 0

    while current_iteration < MAX_TOOL_ITERATIONS:
        current_iteration += 1
        print('Sending to anthropic...')
        print(messages)
        start_time = time.time()
        response_message = client.beta.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219"),
            # Use provided model, fallback if needed
            max_tokens=2048,  # Adjusted from original 20000, tool responses can be large but final answer is one para
            temperature=0.7,  # Adjusted from original 1, for more factual tool use and response
            system=SYSTEM_PROMPT,
            messages=messages,
            tools=anthropic_tools_definition,
            betas=["web-search-2025-03-05"],  # Keep beta for web_search
            tool_choice={"type": "auto"}
        )
        print(f"Took {time.time() - start_time} secs")
        print(response_message)

        if response_message.stop_reason == "tool_use":
            tool_results_for_next_iteration = []
            tool_was_called_by_llm = False

            for content_block in response_message.content:
                if content_block.type == "tool_use":
                    tool_was_called_by_llm = True
                    tool_name = content_block.name
                    tool_input = content_block.input
                    tool_use_id = content_block.id
                    print(f"LLM requests to use tool: {tool_name} with input: {tool_input}")

                    if tool_name == "pharmcat_diplotypes":
                        genes_to_process = tool_input.get("genes")
                        if not isinstance(genes_to_process, list) or not all(
                                isinstance(g, str) for g in genes_to_process):
                            print(
                                f"Error: Invalid 'genes' input for {tool_name}. Expected list of strings. Got: {genes_to_process}")
                            tool_output = {
                                "error": "Invalid input for 'genes'. Expected a list of gene symbols (strings)."}
                        else:
                            tool_output = pharmcat_diplotypes(genes=genes_to_process)

                        tool_results_for_next_iteration.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            # Corrected line below:
                            "content": [{"type": "text", "text": json.dumps(tool_output)}],
                            "is_error": "error" in tool_output  # Optional: flag if result is an error
                        })
                    # Add elif for other custom tools here if any
                    else:
                        # This case handles if the LLM tries to call a tool we haven't explicitly defined a handler for
                        # (e.g., if it hallucinates a tool name or calls web_search in a way that expects us to handle it)
                        print(
                            f"Warning: LLM requested unhandled tool '{tool_name}' or a tool that should be auto-handled by Anthropic.")
                        # We can provide a generic error or attempt to let Anthropic handle it if it's a known type
                        tool_results_for_next_iteration.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": [{"type": "text",
                                         "text": f"Tool '{tool_name}' is not available or handled by this application."}],
                            "is_error": True
                        })

            if not tool_was_called_by_llm:
                # This means stop_reason was 'tool_use', but no tool_use blocks were in content.
                # This is unusual. The LLM might be trying to provide a text response.
                print(
                    "Warning: stop_reason is 'tool_use' but no tool_use content blocks found. Attempting to extract text.")
                text, sources = extract_text_with_citations_from_sdk_blocks(response_message.content)
                if text:  # If there's text, assume it's the final answer
                    return ChatResponse(text=text, sources=sources)
                else:  # No text, and no tool to call, this is a dead end for this iteration.
                    messages.append({"role": "assistant",
                                     "content": response_message.content})  # Add assistant's (empty/confusing) turn
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text",
                                     "text": "No valid tool was called and no text was returned. Please provide a textual answer or call a valid tool."}]
                    })
                    # Loop will continue, hopefully LLM corrects.
            elif tool_results_for_next_iteration:
                # Add assistant's turn (that includes the tool_call requests)
                messages.append({"role": "assistant", "content": response_message.content})
                # Add user's turn (that includes the tool_results)
                messages.append({"role": "user", "content": tool_results_for_next_iteration})
                # Continue the loop for the model to process tool results
            else:
                # Tool was called, but no results were prepared (e.g. unhandled tool name).
                # This state should ideally be covered by the unhandled tool logic above.
                # If we reach here, it's an unexpected state. Break and return error.
                print(
                    f"Error: Tool use indicated but no results prepared. Last assistant message: {response_message.content}")
                return ChatResponse(text="Error processing tool request from the model.", sources=[])


        elif response_message.stop_reason == "end_turn" or any(
                b.type == "text" for b in response_message.content if hasattr(b, 'type')):
            # Model provides a direct textual answer (potentially after using web_search internally)
            text, sources = extract_text_with_citations_from_sdk_blocks(response_message.content)
            return ChatResponse(text=text, sources=sources)

        else:  # Other stop reasons like max_tokens, stop_sequence
            print(f"LLM stopped for reason: {response_message.stop_reason}. Content: {response_message.content}")
            text, sources = extract_text_with_citations_from_sdk_blocks(
                response_message.content)  # Try to get any partial text
            if text:
                return ChatResponse(
                    text=f"{text}\n(Note: LLM processing may have stopped prematurely: {response_message.stop_reason})",
                    sources=sources)
            return ChatResponse(
                text=f"LLM processing stopped: {response_message.stop_reason}. No complete answer available.",
                sources=[])

    # If loop finishes due to MAX_TOOL_ITERATIONS
    print(f"Max tool iterations ({MAX_TOOL_ITERATIONS}) reached. Returning last known text or error.")
    # Try to extract text from the very last response_message, if any was produced before loop termination
    if 'response_message' in locals() and response_message:
        text, sources = extract_text_with_citations_from_sdk_blocks(response_message.content)
        if text:
            return ChatResponse(text=text, sources=sources)
    return ChatResponse(text="The conversation involved too many steps with internal tools and could not be completed.",
                        sources=[])

 
@app.post("/get_snp_base_pairs", response_model=BasePairsResponse)
async def get_snp_base_pairs(request: SNPLocation):
    target_chrom = f"chr{request.chromosome}"
    target_pos = request.position

    reader = vcfpy.Reader.from_path(
        r"/Users/albertcai/Downloads/DuRant_nucleus_dna_download_vcf_NU-DODC-8148.vcf"
    )

    response = ""
    # Search line by line for matching record
    for record in reader:
        if record.CHROM == target_chrom and record.POS == target_pos:
            response += f"Found SNP at {target_chrom}:{target_pos}"

            call = record.calls[0]
            gt = call.data.get("GT")

            # Translate GT to base pairs
            alleles = [record.REF] + [alt.value for alt in record.ALT]
            gt_indices = gt.replace("|", "/").split("/")
            gt_bases = [alleles[int(i)] if i != "." else "." for i in gt_indices]

            response += f"\nGenotype: {gt} → Bases: {'/'.join(gt_bases)}"

            return {"response": response}


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    # Ensure VCF_FILE_PATH is set
    vcf_file = os.environ.get("VCF_FILE_PATH")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not anthropic_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        print("Please create a .env file with ANTHROPIC_API_KEY='your_key' or set it in your environment.")
        exit(1)

    if not vcf_file:
        print("Error: VCF_FILE_PATH environment variable is not set.")
        print("Please create a .env file with VCF_FILE_PATH=/path/to/your/file.vcf or set it in your environment.")
        exit(1)
    if not os.path.exists(vcf_file):
        print(f"Error: VCF file specified by VCF_FILE_PATH does not exist: {vcf_file}")
        exit(1)
    if not os.path.isfile(vcf_file):
        print(f"Error: VCF_FILE_PATH specified is not a file: {vcf_file}")
        exit(1)

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
