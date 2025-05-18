import json
import os
import subprocess
import time

# from functools import lru_cache # Replaced by diskcache
from typing import Annotated, Any, Dict, List

import diskcache  # Added for persistent caching
import requests  # Added for the new tool
from dotenv import load_dotenv
from starlette.responses import FileResponse

cache = diskcache.Cache("pharmcat_cache3")  # Initialize cache

load_dotenv()

import anthropic
import uvicorn
import vcfpy
from fastapi import FastAPI, Header, HTTPException
from fastapi import status as http_status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


# --- PharmCAT diplotypes function with disk caching ---
@cache.memoize()
def pharmcat_diplotypes(genes: List[str]) -> Dict:
    """
    1. Mounts the directory of `vcf_path` into /data in the PharmCAT container.
    2. Runs the full pipeline with the -reporterCallsOnlyTsv flag.
    3. Finds the <basename>.report.tsv and parses out only the requested genes.
    NOTE: This function reads VCF_FILE_PATH from the environment. If this path changes,
    the cache (keyed only by `genes`) might become stale. Clear 'pharmcat_cache' directory
    manually if VCF_FILE_PATH is updated.
    """
    vcf_path = os.environ.get("VCF_FILE_PATH")
    if not vcf_path:
        return {
            "error": "VCF_FILE_PATH environment variable not set.",
            "diplotypes": {},
            "docker_command": "",
        }

    workdir = os.path.dirname(vcf_path)
    base = os.path.basename(vcf_path)
    stem = os.path.splitext(base)[0]

    # Ensure sudo is only prefixed once if already present in the "docker" part of the command
    docker_cmd_parts = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{workdir}:/data",
        "pgkb/pharmcat",
        "pharmcat_pipeline",
        "-reporterCallsOnlyTsv",
        f"/data/{base}",
    ]
    if os.geteuid() == 0:  # if running as root, sudo is not needed
        cmd = docker_cmd_parts
    else:
        cmd = ["sudo"] + docker_cmd_parts

    try:
        process_result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        return {
            "error": f"PharmCAT Docker execution failed: {e.stderr}",
            "diplotypes": {},
            "docker_command": " ".join(cmd),
        }
    except FileNotFoundError:
        return {
            "error": "Docker command (or sudo) not found. Please ensure Docker is installed and in your PATH, and sudo is available if needed.",
            "diplotypes": {},
            "docker_command": " ".join(cmd),
        }

    tsv_path = os.path.join(workdir, f"{stem}.report.tsv")
    if not os.path.exists(tsv_path):
        return {
            "error": f"PharmCAT report TSV file not found: {tsv_path}. PharmCAT may not have generated the expected output. Docker logs (if any during run): {process_result.stderr if 'process_result' in locals() else 'N/A'}",
            "diplotypes": {},
            "docker_command": " ".join(cmd),
        }

    diplotypes = {}
    try:
        with open(tsv_path) as f:
            header_line = ""
            for _ in range(5):
                line = f.readline()
                if not line:
                    break
                if not line:
                    break
                if "Gene" in line and "Source Diplotype" in line:
                    header_line = line
                    break

            if not header_line:
                return {
                    "error": f"PharmCAT report TSV file ({tsv_path}) is missing the expected header ('Gene', 'Source Diplotype').",
                    "diplotypes": {},
                    "docker_command": " ".join(cmd),
                }

            header = header_line.strip().split("\t")
            try:
                gene_idx = header.index("Gene")
                dip_idx = header.index("Source Diplotype")
            except ValueError:
                return {
                    "error": f"PharmCAT report TSV file ({tsv_path}) is missing 'Gene' or 'Source Diplotype' column in the identified header.",
                    "diplotypes": {},
                    "docker_command": " ".join(cmd),
                }

            for line in f:
                cols = line.strip().split("\t")
                if len(cols) > max(gene_idx, dip_idx):
                    if cols[gene_idx] in genes:
                        diplotypes[cols[gene_idx]] = cols[dip_idx]
    except Exception as e:
        return {
            "error": f"Error parsing PharmCAT TSV report '{tsv_path}': {str(e)}",
            "diplotypes": {},
            "docker_command": " ".join(cmd),
        }
    return {"diplotes": diplotypes, "docker_command": " ".join(cmd)}


# --- Tool function for getting SNP base pairs from VCF ---
def get_snp_base_pairs_tool(chromosome: int, position: int) -> Dict[str, Any]:
    """
    Reads a VCF file specified by the VCF_FILE_PATH environment variable
    and retrieves the reference and alternate alleles for a SNP at a given
    chromosome and position. It also determines the genotype.
    """
    vcf_file_path = os.environ.get("VCF_FILE_PATH")
    if not vcf_file_path:
        return {"error": "VCF_FILE_PATH environment variable not set."}
    if not os.path.exists(vcf_file_path):
        return {"error": f"VCF file not found at path: {vcf_file_path}"}

    target_chrom = f"chr{chromosome}"
    if str(chromosome).lower().startswith("chr"):
        target_chrom = str(chromosome)
    target_pos = position

    try:
        reader = vcfpy.Reader.from_path(vcf_file_path)
    except Exception as e:
        return {"error": f"Failed to read VCF file {vcf_file_path}: {str(e)}"}

    response = ""
    # Search line by line for matching record
    for record in reader:
        # print(target_chrom, target_pos)
        # print(record.CHROM, record.POS)

        if record.CHROM == target_chrom and record.POS >= target_pos:
            response += f"Found SNP at {target_chrom}:{target_pos}"

            call = record.calls[0]
            gt = call.data.get("GT")

            # Translate GT to base pairs
            alleles = [record.REF] + [alt.value for alt in record.ALT]
            gt_indices = gt.replace("|", "/").split("/")
            gt_bases = [alleles[int(i)] if i != "." else "." for i in gt_indices]

            response += f"\nGenotype: {gt} → Bases: {'/'.join(gt_bases)}"

            return {"response": response}

    reader.close()
    return {"message": f"SNP not found at {target_chrom}:{target_pos}"}


# --- New tool function for getting SNP info from ClinicalTables API ---
@cache.memoize()  # Cache results for this tool as well
def get_snp_info_from_clinicaltables_tool(rsid: str) -> Dict[str, Any]:
    """
    Queries the NIH Clinical Tables API for a given rsID to get SNP information
    like chromosome, position, alleles, and gene.
    Example API response for rs4988235: [1,["rs4988235"],null,[["rs4988235","2","135851075","G/A, G/C","MCM6"]]]
    """
    if not rsid or not isinstance(rsid, str) or not rsid.startswith("rs"):
        return {
            "error": "Invalid rsID format. It should be a string starting with 'rs' (e.g., 'rs12345')."
        }

    api_url = f"https://clinicaltables.nlm.nih.gov/api/snps/v3/search?terms={rsid}"
    try:
        response = requests.get(api_url, timeout=10)  # Added timeout
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        data = response.json()

        # Expected response structure: [count, [rsid_found], null, [[rsid, chrom, pos, alleles, gene]]]
        if not data or not isinstance(data, list) or len(data) < 4:
            return {
                "error": f"Unexpected API response structure for {rsid} from ClinicalTables."
            }

        count = data[0]
        if count == 0 or not data[3]:  # No results or empty data field
            return {"message": f"SNP {rsid} not found in ClinicalTables database."}

        # Assuming the first result is the one we want if multiple are somehow returned for a specific rsid
        snp_details_list = data[3]
        if (
            not snp_details_list
            or not isinstance(snp_details_list, list)
            or not snp_details_list[0]
        ):
            return {
                "error": f"SNP data field is empty or malformed for {rsid} in ClinicalTables response."
            }

        # Iterate through results to find the exact rsid match, as the API might return related terms
        found_snp = None
        for snp_info_list in snp_details_list:
            if (
                isinstance(snp_info_list, list)
                and len(snp_info_list) >= 5
                and snp_info_list[0].lower() == rsid.lower()
            ):
                found_snp = {
                    "rsid": snp_info_list[0],
                    "chromosome": snp_info_list[1],
                    "position": int(snp_info_list[2]),  # position is typically integer
                    "observed_alleles": snp_info_list[3],
                    "gene": snp_info_list[4]
                    if len(snp_info_list) > 4
                    else "N/A",  # Gene might not always be present
                }
                break

        if found_snp:
            return {"snp_info": found_snp}
        else:
            return {
                "message": f"SNP {rsid} not found or data format incorrect in ClinicalTables response details."
            }

    except requests.exceptions.HTTPError as http_err:
        return {
            "error": f"HTTP error occurred while querying ClinicalTables for {rsid}: {http_err}"
        }
    except requests.exceptions.ConnectionError as conn_err:
        return {
            "error": f"Connection error occurred while querying ClinicalTables for {rsid}: {conn_err}"
        }
    except requests.exceptions.Timeout as timeout_err:
        return {
            "error": f"Timeout occurred while querying ClinicalTables for {rsid}: {timeout_err}"
        }
    except requests.exceptions.RequestException as req_err:
        return {
            "error": f"An error occurred while querying ClinicalTables for {rsid}: {req_err}"
        }
    except ValueError as json_err:  # Includes json.JSONDecodeError
        return {
            "error": f"Failed to decode JSON response from ClinicalTables for {rsid}: {json_err}"
        }


# --- Pydantic Models ---
class ChatRequest(BaseModel):
    messages: List[Dict[str, Any]]


class ChatResponse(BaseModel):
    messages: List[Dict[str, Any]]


# --- User's original citation extraction function ---
def extract_text_with_citations_from_sdk_blocks(api_response_content_blocks: List[Any]):
    user_texts = []
    citations_list = []
    # ... (rest of the function remains unchanged)
    for block_sdk_obj in api_response_content_blocks:
        block_dict = {}
        if hasattr(block_sdk_obj, "type"):
            if block_sdk_obj.type == "text":
                text_content = getattr(block_sdk_obj, "text", "").strip()
                if text_content:
                    user_texts.append(text_content)
            try:
                block_dict = dict(block_sdk_obj)
            except Exception:
                block_dict = {}
        elif isinstance(block_sdk_obj, dict):
            block_dict = block_sdk_obj
            if block_dict.get("type") == "text":
                text_content = block_dict.get("text", "").strip()
                if text_content and text_content not in user_texts:
                    user_texts.append(text_content)

        if block_dict.get("citations"):
            for cite_obj in block_dict.get("citations", []):
                cite_dict = {}
                try:
                    cite_dict = dict(cite_obj)
                except Exception:
                    if hasattr(cite_obj, "url") and hasattr(cite_obj, "title"):
                        cite_dict = {
                            "url": getattr(cite_obj, "url"),
                            "title": getattr(cite_obj, "title"),
                        }
                    else:
                        continue
                url = cite_dict.get("url")
                title = cite_dict.get("title")
                if url and title:
                    citations_list.append(f"{title}: {url}")
    full_text = "\n\n".join(user_texts)
    return full_text, list(set(citations_list))


# --- FastAPI Endpoints ---
@app.post("/chat")
async def search_snpedia(
    request: ChatRequest,
    authorization: Annotated[str | None, Header()] = None,
):  # Updated request model
    token = authorization.removeprefix("Bearer ").strip() if authorization else None
    if token != os.environ.get(
        "AUTHORIZATION_SECRET"
    ):  # Consider making token configurable via env var
        raise HTTPException(
            status_code=http_status.HTTP_401_UNAUTHORIZED, detail="Unauthorized"
        )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    SYSTEM_PROMPT = """
    You are a helpful assistant that provides information about SNPs and their effects on health.
    The user will provide their messages, which may or may not include their SNPs. They will ask questions and you should provide personalized health advice.
    If you need to know the user's specific SNPs to answer a question, you should ask them or look for them in the conversation history.
    You should also provide sources where the user can get more information if needed. Omit any information not relevant to the specific user question.
    Use simple language with direct answers.
    Responses should be a maximum of one paragraph unless more detail is clearly requested or necessary.
    Try your best to not ask follow up questions, if not absolutely necessary.

    You have access to the following tools:
    1. `web_search`: To search SNPedia.com for information about SNPs. Use this to answer questions about specific SNPs if detailed information from SNPedia is required.
    2. `pharmcat_diplotypes`: To run the PharmCAT Docker pipeline on the user's VCF data (implicitly available to the tool) and return a mapping of gene to star-allele diplotype for requested genes. Use this tool when the user's question pertains to pharmacogenomics, drug metabolism (e.g. for genes like CYP2C19), or requires diplotype information for specific genes mentioned or implied in the question. Only specify genes relevant to the user's question in the tool input. The VCF file is pre-configured on the server.
    3. `get_snp_base_pairs`: To retrieve detailed information (REF/ALT alleles, genotype) for a specific SNP from the user's VCF file, given its chromosome and position. Use this if you need to know the exact genetic variation at a specific locus from the VCF. The VCF file is pre-configured on the server.
    4. `get_snp_info_from_clinicaltables`: To retrieve SNP information (chromosome, position, observed alleles, gene) for a given rsID from the NIH Clinical Tables API. Use this if you need general information about an rsID like its genomic location or associated gene.
    """
    messages = [dict(msg) for msg in request.messages]

    anthropic_tools_definition = [
        {
            "name": "web_search",
            "type": "web_search_20250305",  # Use the beta type
            "allowed_domains": ["snpedia.com"],
        },
        {
            "name": "pharmcat_diplotypes",
            "description": "Run the PharmCAT Docker pipeline on the user's VCF and return a mapping of gene→star‑allele diplotype for requested genes. The VCF file is pre-configured on the server.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "genes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of genes to process (e.g., GBA, LRRK2, CYP2C19). Only include genes relevant to the user's question.",
                    }
                },
                "required": ["genes"],
            },
        },
        {
            "name": "get_snp_base_pairs",  # This tool was previously commented out, re-enabling it.
            "description": "Retrieves REF/ALT alleles and genotype for a SNP from the user's VCF file given its chromosome and position. The VCF file is pre-configured on the server.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "chromosome": {
                        "type": "integer",
                        "description": "The chromosome number (e.g., 1, 22). Do not include 'chr'.",
                    },
                    "position": {
                        "type": "integer",
                        "description": "The base pair position of the SNP.",
                    },
                },
                "required": ["chromosome", "position"],
            },
        },
        {  # Definition for the new tool
            "name": "get_snp_info_from_clinicaltables",
            "description": "Retrieves SNP information (chromosome, position, observed alleles, gene) for a given rsID from the NIH Clinical Tables API.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "rsid": {
                        "type": "string",
                        "description": "The rsID of the SNP to look up (e.g., 'rs4988235').",
                    }
                },
                "required": ["rsid"],
            },
        },
    ]

    MAX_TOOL_ITERATIONS = 5
    current_iteration = 0

    async def process_stream():
        nonlocal current_iteration, messages

        while current_iteration < MAX_TOOL_ITERATIONS:
            current_iteration += 1
            print(f"Iteration {current_iteration}. Sending to Anthropic...")

            start_time = time.time()
            collected_content = []
            response_message = None

            with client.beta.messages.stream(
                model=os.environ.get("ANTHROPIC_MODEL", "claude-3-7-sonnet-20250219"),
                max_tokens=2048,
                temperature=0.7,
                system=SYSTEM_PROMPT,
                messages=messages,
                tools=anthropic_tools_definition,
                betas=["web-search-2025-03-05"],
                tool_choice={"type": "auto"},
            ) as stream:
                for event in stream:
                    # Deep conversion of all nested objects to dictionaries
                    def convert_to_dict(obj):
                        if hasattr(obj, "__dict__"):
                            return {
                                k: convert_to_dict(v)
                                for k, v in obj.__dict__.items()
                                if not k.startswith("_")
                            }
                        elif isinstance(obj, dict):
                            return {k: convert_to_dict(v) for k, v in obj.items()}
                        elif isinstance(obj, list):
                            return [convert_to_dict(i) for i in obj]
                        else:
                            return obj

                    try:
                        event_dict = dict(event)

                        for key in event_dict:
                            if hasattr(event_dict[key], "__dict__") or isinstance(
                                event_dict[key], (list, dict)
                            ):
                                event_dict[key] = convert_to_dict(event_dict[key])

                        print(f"Event: {event_dict}")

                        yield json.dumps(event_dict) + "\n"

                    except (TypeError, ValueError) as e:
                        print(
                            f"Error converting event to dict: {e}, attempting deep conversion"
                        )
                        event_dict = convert_to_dict(event)
                        print(f"Deep converted event: {event_dict}")
                        yield json.dumps(event_dict) + "\n"

                    # Collect response for further processing
                    if "message" in event_dict:
                        response_message = event_dict["message"]
                    if "content_block" in event_dict:
                        collected_content.append(event_dict["content_block"])

            if not response_message:
                response_message = {
                    "content": collected_content,
                    "stop_reason": "end_turn",
                }

            # Handle tool use cases
            if response_message.get("stop_reason") == "tool_use":
                tool_results_for_next_iteration = []
                tool_was_called_by_llm = False

                for content_block in response_message.get("content", []):
                    if content_block.get("type") == "tool_use":
                        tool_was_called_by_llm = True
                        tool_name = content_block.get("name")
                        tool_input = content_block.get("input")
                        tool_use_id = content_block.get("id")
                        print(
                            f"LLM requests to use tool: {tool_name} with input: {tool_input}"
                        )

                        tool_output_content = (
                            None  # Will hold the JSON string for the tool result
                        )

                        if tool_name == "pharmcat_diplotypes":
                            genes_to_process = tool_input.get("genes")
                            if not isinstance(genes_to_process, list) or not all(
                                isinstance(g, str) for g in genes_to_process
                            ):
                                print(
                                    f"Error: Invalid 'genes' input for {tool_name}. Expected list of strings. Got: {genes_to_process}"
                                )
                                tool_output = {
                                    "error": "Invalid input for 'genes'. Expected a list of gene symbols (strings)."
                                }
                            else:
                                # Convert to tuple for caching if genes list can be large, though diskcache handles lists
                                tool_output = pharmcat_diplotypes(
                                    genes=tuple(sorted(genes_to_process))
                                )
                            print(f"{tool_output=}")
                            tool_output_content = json.dumps(tool_output)

                        elif tool_name == "get_snp_base_pairs":
                            chromosome = tool_input.get("chromosome")
                            position = tool_input.get("position")
                            if not isinstance(chromosome, int) or not isinstance(
                                position, int
                            ):
                                print(
                                    f"Error: Invalid input for {tool_name}. Expected integer chromosome and position. Got: chr={chromosome}, pos={position}"
                                )
                                tool_output = {
                                    "error": "Invalid input. 'chromosome' and 'position' must be integers."
                                }
                            else:
                                tool_output = get_snp_base_pairs_tool(
                                    chromosome=chromosome, position=position
                                )
                            print(f"{tool_output=}")
                            tool_output_content = json.dumps(tool_output)

                        elif tool_name == "get_snp_info_from_clinicaltables":
                            rsid_input = tool_input.get("rsid")
                            if not isinstance(
                                rsid_input, str
                            ) or not rsid_input.startswith("rs"):
                                print(
                                    f"Error: Invalid 'rsid' input for {tool_name}. Expected string like 'rs123'. Got: {rsid_input}"
                                )
                                tool_output = {
                                    "error": "Invalid input for 'rsid'. Expected a string starting with 'rs'."
                                }
                            else:
                                tool_output = get_snp_info_from_clinicaltables_tool(
                                    rsid=rsid_input
                                )
                            print(f"ClinicalTables SNP info output: {tool_output}")
                            tool_output_content = json.dumps(tool_output)

                        else:  # web_search or unhandled tool
                            print(
                                f"Warning: LLM requested tool '{tool_name}' which might be auto-handled or is unhandled by custom logic."
                            )
                            tool_results_for_next_iteration.append(
                                {
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": [
                                        {
                                            "type": "text",
                                            "text": f"Tool '{tool_name}' is not one of the custom-handled tools by this application or is an auto-handled tool like web_search.",
                                        }
                                    ],
                                    "is_error": True,
                                }
                            )
                            continue

                        # Common way to append tool result
                        tool_results_for_next_iteration.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tool_use_id,
                                "content": [
                                    {"type": "text", "text": tool_output_content}
                                ],
                                "is_error": "error" in tool_output
                                if isinstance(tool_output, dict)
                                else False,
                            }
                        )

                if not tool_was_called_by_llm:
                    # No tools were called - check if text is available
                    if any(
                        b.get("type") == "text"
                        for b in response_message.get("content", [])
                    ):
                        print(
                            "Text found instead of tool use. Returning as final answer."
                        )
                        return

                    # If no text and no tool use, add a message to try again
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_message.get("content", []),
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "No valid tool was called and no text was returned. Please provide a textual answer or call a valid tool.",
                                }
                            ],
                        }
                    )
                elif tool_results_for_next_iteration:
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_message.get("content", []),
                        }
                    )
                    messages.append(
                        {"role": "user", "content": tool_results_for_next_iteration}
                    )
                    # Yield a message about tool use
                    yield (
                        json.dumps(
                            {
                                "type": "info",
                                "text": f"Using tools: {', '.join(t.get('type') for t in tool_results_for_next_iteration)}",
                            }
                        )
                        + "\n"
                    )
                else:
                    # Tool use indicated but nothing processed
                    messages.append(
                        {
                            "role": "assistant",
                            "content": response_message.get("content", []),
                        }
                    )
                    messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": "An error occurred while trying to use a tool. Please try to answer directly or use a different tool.",
                                }
                            ],
                        }
                    )
            else:
                # This is a final answer (not tool use)
                return

        # Max iterations reached
        yield (
            json.dumps(
                {
                    "type": "error",
                    "text": f"Max tool iterations ({MAX_TOOL_ITERATIONS}) reached. The conversation involved too many tool steps.",
                }
            )
            + "\n"
        )

    return StreamingResponse(process_stream(), media_type="application/x-ndjson")


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.get("/")
async def main():
    return FileResponse("ui.html")


if __name__ == "__main__":
    vcf_file = os.environ.get("VCF_FILE_PATH")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")

    if not anthropic_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        exit(1)
    if not vcf_file:
        print(
            "Warning: VCF_FILE_PATH environment variable is not set. Tools 'pharmcat_diplotypes' and 'get_snp_base_pairs' will fail if used."
        )
        # Allowing to run if other tools like clinicaltables or web_search are to be tested without VCF
    elif not os.path.exists(vcf_file):
        print(f"Error: VCF file specified by VCF_FILE_PATH does not exist: {vcf_file}")
        exit(1)
    elif not os.path.isfile(vcf_file):
        print(f"Error: VCF_FILE_PATH specified is not a file: {vcf_file}")
        exit(1)

    print(
        f"PharmCAT and ClinicalTables API cache directory: {os.path.abspath(cache.directory)}"
    )
    print(
        "To test the new tool, ask the assistant for information about a specific rsID, e.g., 'Tell me about rs4988235'."
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
