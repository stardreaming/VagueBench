import base64
import re
from mimetypes import guess_type


template_lang = '''\
The output json format should be {'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, \
2. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, \
3. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!
'''


template = '''
The output json format should be {'visual_state_description':str, 'reasoning_and_reflection':str, 'language_plan':str, 'executable_plan':List[{'action_id':int, 'action_name':str}...]}
The fields in above JSON follows the purpose below:
1. visual_state_description is for description of current state from the visual image, 
2. reasoning_and_reflection is for summarizing the history of interactions and any available environmental feedback. Additionally, provide reasoning as to why the last action or plan failed and did not finish the task, 
3. language_plan is for describing a list of actions to achieve the user instruction. Each action is started by the step number and the action name, 
4. executable_plan is a list of actions needed to achieve the user instruction, with each action having an action ID and a name.
5. keep your plan efficient and concise.
!!! When generating content for JSON strings, avoid using any contractions or abbreviated forms (like 's, 're, 've, 'll, 'd, n't) that use apostrophes. Instead, write out full forms (is, are, have, will, would, not) to prevent parsing errors in JSON. Please do not output any other thing more than the above-mentioned JSON, do not include ```json and ```!!!.
'''


def fix_json(json_str):
    """
    Locates the substring between the keys "reasoning_and_reflection" and "language_plan"
    and escapes any inner double quotes that are not already escaped.
    
    The regex uses a positive lookahead to stop matching when reaching the delimiter for the next key.
    """
    # first fix common errors
    json_str = json_str.replace("'",'"') 
    json_str = json_str.replace('\"s ', "\'s ")
    json_str = json_str.replace('\"re ', "\'re ")
    json_str = json_str.replace('\"ll ', "\'ll ")
    json_str = json_str.replace('\"t ', "\'t ")
    json_str = json_str.replace('\"d ', "\'d ")
    json_str = json_str.replace('\"m ', "\'m ")
    json_str = json_str.replace('\"ve ', "\'ve ")
    json_str = json_str.replace('```json', '').replace('```', '')

    # Then fix some situations. Pattern explanation:
    # 1. ("reasoning_and_reflection"\s*:\s*") matches the key and the opening quote.
    # 2. (?P<value>.*?) lazily captures everything in a group named 'value'.
    # 3. (?=",\s*"language_plan") is a positive lookahead that stops matching before the closing quote
    #    that comes before the "language_plan" key.
    pattern = r'("reasoning_and_reflection"\s*:\s*")(?P<value>.*?)(?=",\s*"language_plan")'
    
    def replacer(match):
        prefix = match.group(1)            # Contains the key and the opening quote.
        value = match.group("value")         # The raw value that might contain unescaped quotes.
        # Escape any double quote that is not already escaped.
        fixed_value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + fixed_value

    # Use re.DOTALL so that newlines in the value are included.
    fixed_json = re.sub(pattern, replacer, json_str, flags=re.DOTALL)
    return fixed_json


def convert_format_2claude(messages):
    new_messages = []
    
    for message in messages:
        if message["role"] == "user":
            new_content = []
    
            for item in message["content"]:
                if item.get("type") == "image_url":
                    base64_data = item["image_url"]["url"][22:]
                    new_item = {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": base64_data
                        }
                    }
                    new_content.append(new_item)
                else:
                    new_content.append(item)

            new_message = message.copy()
            new_message["content"] = new_content
            new_messages.append(new_message)

        else:
            new_messages.append(message)

    return new_messages


def local_image_to_data_url(image_path):
    # Guess the MIME type of the image based on the file extension
    mime_type, _ = guess_type(image_path)
    if mime_type is None:
        mime_type = 'application/octet-stream'  # Default MIME type if none is found

    # Read and encode the image file
    with open(image_path, "rb") as image_file:
        base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

    # Construct the data URL
    return f"data:{mime_type};base64,{base64_encoded_data}"
