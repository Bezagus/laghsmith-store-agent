"""Utilidades para integración con Gemini API"""
from google.genai import types


def convert_to_gemini_content(messages):
    """Convierte mensajes formato OpenAI a Gemini Content"""
    contents = []
    for msg in messages:
        if msg["role"] == "system":
            continue  # Sistema se pasa por separado
        elif msg["role"] == "user":
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text=msg["content"])]
            ))
        elif msg["role"] == "assistant":
            if msg.get("content"):
                contents.append(types.Content(
                    role="model",
                    parts=[types.Part(text=msg["content"])]
                ))
            elif msg.get("function_calls"):
                parts = []
                for fc in msg["function_calls"]:
                    parts.append(types.Part(
                        function_call=types.FunctionCall(
                            name=fc.name,
                            args=fc.args
                        )
                    ))
                contents.append(types.Content(role="model", parts=parts))
        elif msg["role"] == "tool":
            contents.append(types.Content(
                role="user",
                parts=[types.Part(
                    function_response=types.FunctionResponse(
                        name=msg["name"],
                        response={"result": msg["content"]}
                    )
                )]
            ))
    return contents


def extract_system_instruction(messages):
    """Extrae el mensaje de sistema de la lista de mensajes"""
    for msg in messages:
        if msg["role"] == "system":
            return msg["content"]
    return None


def create_gemini_tools(openai_tools):
    """Convierte definición de tools de OpenAI a Gemini"""
    function_declarations = []
    for tool in openai_tools:
        if tool["type"] == "function":
            func = tool["function"]
            function_declarations.append(
                types.FunctionDeclaration(
                    name=func["name"],
                    description=func["description"],
                    parameters=func["parameters"]
                )
            )
    return types.Tool(function_declarations=function_declarations)
