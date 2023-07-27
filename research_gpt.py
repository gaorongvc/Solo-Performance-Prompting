from models import OpenAIWrapper

from run import default_gpt_config, _post_process_raw_response
from tasks import get_task

task_name = 'research'
task = get_task(task_name, file='research_1.jsonl')
i = 0
method = 'standard'
num_generation = 1
SYSTEM_MESSAGE = "You are an AI assistant that helps people find information."  # or "" (empty string)
config = default_gpt_config
config['model'] = 'gpt-4-0613'
gpt = OpenAIWrapper(config=config, system_message=SYSTEM_MESSAGE)

prompt = task.get_input_prompt(i, method=method)
# get raw response
raw_output_batch, raw_response_batch = gpt.run(prompt=prompt, n=num_generation)
if raw_output_batch == [] or raw_response_batch == []:  # handle exception
    exit()
    # get parsed response, and the success flags (whether or not the parsing is success) (standard prompt always success)
unwrapped_output_batch, if_success_batch = _post_process_raw_response(task, raw_output_batch, method)
print(unwrapped_output_batch)
