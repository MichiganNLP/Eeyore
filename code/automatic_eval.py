import argparse
import json
import asyncio
from openai import OpenAI, AsyncOpenAI
import copy
from string import Template
import logging
from constant import interview_questions, after_interview_questions, judger_prompt, system_prompt_template
from roleplay_doh import roleplay_doh_rewrite_response
from patient_psi import create_cognitive_system_prompt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def prepare_prompt_from_profile(data=None):
    prof = data["profile"]
    name_tb = prof.get("name", "").lower()
    age_tb = prof.get("age", "").lower()
    gender_dd = prof.get("gender", "").lower()
    occp_tb = prof.get("occupation", "").lower()
    marital_dd = prof.get("marital status", "").lower()
    sit_tb = prof.get("situation of the client", "").lower()
    history_tb = prof.get("counseling history", "")
    resis_cb = prof.get("resistance toward the support", "").lower()

    mild_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "mild" in v.lower()]
    mod_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "moderate" in v.lower()]
    seve_sym_dd = [k.lower() for k, v in prof.get("symptom severity", {}).items() if "severe" in v.lower()]

    mild_cog_dd = [k.lower() for k, v in prof.get("cognition distortion exhibition", {}).items() if "not exhibited" not in v.lower()]
    mod_cog_dd = []
    seve_cog_dd = []

    overall_dd = prof.get("depression severity", "")
    suicidal_dd = prof.get("suicidal ideation severity", "")
    homicidal_dd = prof.get("homicidal ideation severity", "")


    return await get_system_prompt_with_profile(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb,resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd)



async def get_system_prompt_with_profile(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb, resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd):   
    """
    This function gets the system prompt with the profile dictionary.
    """
    
    profile_dict = {"name":"", "gender":"", "age":"", "marital_status":"", "occupation":"", "situation_of_the_client":"", "counseling_history":"", "resistance_toward_the_support":"", "symptom_severity":"", "cognition_distortion_exhibition":"", "depression_severity":"", "suicidal_ideation_severity":"", "homicidal_ideation_severity":""}
    system_prompt = await parse_system_prompt(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb, resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd)
    logger.info(f"System Prompt: {system_prompt}")
    patient_profile = "## PROFILE\n" + system_prompt.split("YOUR PROFILE:")[-1].split("YOUR TASK:")[0]
    
    profile_dict["name"] = validate_input(name_tb)
    profile_dict["age"] = validate_input(age_tb)
    profile_dict["gender"] = validate_input(gender_dd)
    profile_dict["occupation"] = validate_input(occp_tb)
    profile_dict["situation_of_the_client"] = validate_input(sit_tb)
    profile_dict["marital_status"] = validate_input(marital_dd)
    profile_dict["resistance_toward_the_support"] = validate_input(resis_cb)
    profile_dict["counseling_history"] = validate_input(history_tb)
    profile_dict["symptom_severity_mild"] = mild_sym_dd
    profile_dict["symptom_severity_moderate"] = mod_sym_dd
    profile_dict["symptom_severity_severe"] = seve_sym_dd
    profile_dict["cognitive_distortion"] = mild_cog_dd
    profile_dict["depression_severity"] = overall_dd
    profile_dict["suicidal_ideation_severity"] = validate_input(suicidal_dd)
    profile_dict["homicidal_ideation_severity"] = validate_input(homicidal_dd)
    
    return system_prompt, patient_profile, profile_dict


def validate_input(input):
    if input is None:
        return ""
    if input == "" or input.lower() == "not specified" or input.lower() == "unknown" or input.lower()=="n/a" or "cannot be identified" in input.lower() or "cannot be determined" in input.lower() or "not mention" in input.lower() or "not exhibited" in input.lower():
        return ""  
    else:
        return input


async def parse_system_prompt(name_tb, age_tb, gender_dd, occp_tb, marital_dd, sit_tb, history_tb,resis_cb, mild_sym_dd, mod_sym_dd, seve_sym_dd, mild_cog_dd, mod_cog_dd, seve_cog_dd, overall_dd, suicidal_dd, homicidal_dd):
    temp_profile_dict = {"name":"", "gender":"", "age":"", "marital_status":"", "occupation":"", "situation_of_the_client":"", "counseling_history":"", "resistance_toward_the_support":"", "symptom_severity":"", "cognitive_distortion":"", "depression_severity":"", "suicidal_ideation_severity":"", "homicidal_ideation_severity":""}
    if validate_input(name_tb):
        temp_profile_dict["name"] = "- " + "name" + ": " + validate_input(name_tb) + "\n"
    if validate_input(age_tb):
        temp_profile_dict["age"] = "- " + "age" + ": " + validate_input(age_tb) + "\n"
    if validate_input(gender_dd):
        temp_profile_dict["gender"] = "- " + "gender" + ": " + validate_input(gender_dd) + "\n"
    if validate_input(occp_tb):
        temp_profile_dict["occupation"] = "- " + "occupation" + ": " + validate_input(occp_tb) + "\n"
    if validate_input(sit_tb):
        temp_profile_dict["situation_of_the_client"] = "- " + "situation of the client" + ": " + validate_input(sit_tb) + "\n"
    if validate_input(marital_dd):
        temp_profile_dict["marital_status"] = "- " + "marital status" + ": " + validate_input(marital_dd) + "\n"
    if validate_input(resis_cb):
        temp_profile_dict["resistance_toward_the_support"] = "- " + "resistance toward the support" + ": " + validate_input(resis_cb) + "\n"
    if validate_input(history_tb):
        temp_profile_dict["counseling_history"] = "- " + "counseling history" + ": " + validate_input(history_tb) + "\n"
    
    sup = ""
    for item in seve_sym_dd:
        sup += "  - " + str(item) + ": " + "severe" + "\n"  
    for item in mod_sym_dd:
        sup += "  - " + str(item) + ": " + "moderate" + "\n" 
    for item in mild_sym_dd:
        sup += "  - " + str(item) + ": " + "mild" + "\n"  
    if sup:
        temp_profile_dict["symptom_severity"] = "- " + "symptom severity" + "\n" + sup


    sup = ""
    for item in seve_cog_dd:
        sup += "  - " + str(item) + ": " + "severe" + "\n"  
    for item in mod_cog_dd:
        sup += "  - " + str(item) + ": " + "moderate" + "\n" 
    for item in mild_cog_dd:
        sup += "  - " + str(item) + ": " + "exhibited" + "\n" 
    if sup: 
        temp_profile_dict["cognition_distortion_exhibition"] = "- " + "cognition distortion exhibition" + "\n" + sup
    
    if validate_input(overall_dd):
        temp_profile_dict["depression_severity"] = "- " + "depression severity" + ": " + validate_input(overall_dd) + "\n"

    if validate_input(suicidal_dd):
        temp_profile_dict["suicidal_ideation_severity"] = "- " + "suicidal ideation severity" + ": " + validate_input(suicidal_dd) + "\n"

    if validate_input(homicidal_dd):
        temp_profile_dict["homicidal_ideation_severity"] = "- " + "homicidal ideation severity" + ": " + validate_input(homicidal_dd) + "\n"
    
    system_prompt = Template(system_prompt_template).safe_substitute(temp_profile_dict) 
    
    return system_prompt

def get_interview_questions(profile_dict, category, trait=None):
    question_list = interview_questions[category]
    for i in range(len(question_list)):
        if "SYMPTOM" in question_list[i]:
            question_list[i] = question_list[i].replace("SYMPTOM", trait)
        if "COGNITIVE DISTORTION" in question_list[i]:
            question_list[i] = question_list[i].replace("COGNITIVE DISTORTION", trait)
    return question_list


async def conduct_interview_for_category(interviewed_agent, initial_prompts, questions_to_ask, q_category, trait, profile, args):
    category_prompts = copy.deepcopy(initial_prompts[:])
    # conduct interview
    for question in questions_to_ask:
        # adding questions to prompt
        category_prompts.append({"role": "user", "content": question})   
        logger.info(f"Asking Question about {q_category} {trait}: {question}")
        response = await interviewed_agent.chat.completions.create(model=args.experiment_model, messages=category_prompts, temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens)
        # if the response is valid
        if response:
            # getting the content
            response_content = response.choices[0].message.content
                        
            if args.experiment_name == "roleplay-doh":
                response_content = await roleplay_doh_rewrite_response(interviewed_agent, initial_prompts, response_content, profile, args)
                
            logger.info(f"AI: {response_content}")
            category_prompts.append({"role": "assistant", "content": response_content})
        else: # printing error message
            logger.info("Failed to generate a response for question.")
            
    return category_prompts

# asks questions for each category
async def get_score_for_category(judger_client, interview_results, eval_questions, q_category, trait, model):
    # if the interview is not valid, retry up to 3 times
    for i in range(3):      
        judger_system_prompt = "You are a helpful AI assistant. Answer the questions I ask you with a singular word: Yes or No based on the text data you are provided with."
        messages = [{"role": "system", "content": judger_system_prompt}]     
        # Prepare interview scripts for evaluation
        survey_prompt = "\n".join([category_prompt["role"]+":"+category_prompt["content"] for category_prompt in interview_results[1:]])
        if "symptom_severity" in q_category:
            trait = "symptom severity: " + trait + "- " + q_category.split("_")[-1]
        elif "cognitive_distortion" in q_category: 
            trait = "cognitive distortion: " + trait + "- " + "exhibited"
        else:
            trait = q_category.replace("_", " ") + ": " + trait
        # Prepare prompt for evaluation
        eval_prompt = Template(judger_prompt).safe_substitute({
            "conv":survey_prompt,
            "trait": trait,
            "question":eval_questions,    
        }) 
        
        logger.info(f"\nAsking Yes/No Question: \n{eval_prompt}")
        messages.append({"role": "user", "content": eval_prompt})
        
        survey_response = await judger_client.chat.completions.create(model=model, messages=messages)
        logger.info("HERE IS THE SURVERY RESPONSE:")
        answer = survey_response.choices[0].message.content
        logger.info(f"ANSWER: {answer}")
        # we update score at the bottom of the main function 
        if answer.lower().strip() in ["1","2","3","4","5"]:
            return int(answer.lower().strip())
        else:
            logger.info("Failed to generate judger response, re-trying...")    
    return -1



def parse_arguments():
    
    """Parse command line arguments for the automatic evaluation script."""
    parser = argparse.ArgumentParser(
        description="Automatic evaluation script for experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # File paths
    parser.add_argument(
        "--profile-path",
        type=str,
        default="./data/test_profile_cognitive_model.json",
        help="Path to the test profile JSON file. Running 12 profile at once may take a while. You may want to use symptom separated profile for efficiency."
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Path to the output directory for results"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="eeyore",
        help="Name of the experiment: roleplay-doh, patient-psi, eeyore"
    )
    
    # OpenAI model configuration
    parser.add_argument(
        "--experiment-model",
        type=str,
        default="eeyore_sft_epoch2_dpo_round1_epoch1_dpo_round2_epoch1_llama3.1_8B",
        help="model name for experiments. For baseline experiments, use gpt-4o-2024-08-06. For our model, use eeyore model name"
    )
    
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://127.0.0.1:6416/v1",
        help="Base URL for deployed model API. For baseline experiments, use https://api.openai.com/v1. For our model, use https://127.0.0.1:6416/v1"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key for baseline experiments"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Temperature for our model generation. For eeyore, let it be undefined"
    )
    
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.8,
        help="Top-p for our model generation. For eeyore, let it be undefined"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Maximum tokens for model generation"
    )
    
    
    parser.add_argument(
        "--openai-api-key",
        type=str,
        help="API key for baseline and judger models"
    )
    
    parser.add_argument(
        "--judger-model",
        type=str,
        default="gpt-4o-2024-08-06",
        help="OpenAI model name for baseline experiments"
    )
    
    # Processing options
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Starting index for processing profiles, in case you want to run a subset of profiles"
    )
    
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Ending index for processing profiles (None for all), in case you want to run a subset of profiles"
    )
    
    return parser.parse_args()



async def main():
    args = parse_arguments()   
    profile_path = args.profile_path
    output_dir = args.output_dir
    
    with open(profile_path, "r") as f:
        test_set = json.load(f)
    
    index = args.start_index
    experiment = args.experiment_name
    
    if not args.openai_api_key:
        args.openai_api_key = args.api_key
    
    # defining the survey agent    
    judger_client = AsyncOpenAI(
        api_key=args.openai_api_key,
        base_url="https://api.openai.com/v1"
    )

    
    for index, data in enumerate(test_set):
        # Apply end_index limit if specified
        if args.end_index is not None and index >= args.end_index:
            break
            
        data.pop("conversation")
        
        
        interviewed_client = AsyncOpenAI(
            base_url=args.base_url,
            api_key=args.api_key,
        )
        model = args.experiment_model
    
        prompts = []
        scores = []
        sys_prompt, profile, profile_dict = await prepare_prompt_from_profile(data)
        
        if experiment == "patient-psi":
            sys_prompt, profile, profile_dict = await create_cognitive_system_prompt(data, profile, profile_dict)
        logger.info(f"System Prompt: {sys_prompt}")
        prompts.append({"role":"system","content":f"{sys_prompt}"})
        scores = {}
        for q_category in interview_questions:
            if q_category in profile_dict and profile_dict[q_category]:
                if "symptom_severity" in q_category or "cognitive_distortion" in q_category:
                    for trait in profile_dict[q_category]:
                        # Get interview questions
                        questions_to_ask = get_interview_questions(profile, q_category, trait)
                        # Get evaluation questions after the interview
                        eval_questions = after_interview_questions[q_category]
                        if q_category+":"+trait not in scores:
                            scores[q_category+":"+trait] = 0
                        # conduct interview for each category
                        interview_results = await conduct_interview_for_category(interviewed_client, prompts, questions_to_ask, q_category, trait, profile, args)
                        # get score for each category
                        scores[q_category+":"+trait] += await get_score_for_category(judger_client, interview_results, eval_questions, q_category, trait, args.judger_model)
                else:
                    questions_to_ask = get_interview_questions(profile, q_category)
                    trait = profile_dict[q_category]
                    if q_category+":"+trait not in scores:
                        scores[q_category+":"+trait] = 0
                    eval_questions = after_interview_questions[q_category]
                    interview_results = await conduct_interview_for_category(interviewed_client, prompts, questions_to_ask, q_category, trait, profile, args)
                    scores[q_category+":"+trait] += await get_score_for_category(judger_client, interview_results, eval_questions, q_category, trait, args.judger_model)
            else:
                logger.info(q_category + " not in profile:" + str(profile_dict.keys()))
        data["roleplay-doh"] = scores
        with open(os.path.join(output_dir, experiment+"_output.json"), "w", encoding="utf-8") as f:
            json.dump(test_set, f, indent=1)
            
if __name__ == "__main__":
    asyncio.run(main())