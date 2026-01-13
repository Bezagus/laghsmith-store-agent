from google import genai
from google.genai import types
from langsmith import Client, traceable
from pipeline import initialize_messages, tools
from gemini_utils import convert_to_gemini_content, extract_system_instruction
from agent_tools import (
    calcular_precio,
    buscar_productos,
    sumar_precios,
    verificar_descuento
)
from dotenv import load_dotenv
import json
import os

load_dotenv()
gemini_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

@traceable
def target(inputs: dict) -> dict:
    try:
        print("\n=== TARGET FUNCTION ===")
        print(f"Input recibido: {inputs}")
        print(f"Tipo de input: {type(inputs)}")
        
        messages = initialize_messages()
        print(f"Mensajes iniciales: {messages}")
        
        question = inputs.get("question", "") if isinstance(inputs, dict) else inputs
        print(f"Pregunta procesada: {question}")
        
        messages.append({"role": "user", "content": question})
        print(f"Mensajes finales: {messages}")
        
        while True:
            try:
                # Preparar la llamada a Gemini
                system_instruction = extract_system_instruction(messages)
                contents = convert_to_gemini_content([m for m in messages if m["role"] != "system"])

                config = types.GenerateContentConfig(
                    tools=[tools],
                    system_instruction=system_instruction
                )

                response = gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=contents,
                    config=config
                )

                # Parsear respuesta de Gemini
                candidate = response.candidates[0]
                parts = candidate.content.parts

                text_response = None
                function_calls = []

                for part in parts:
                    if part.text:
                        text_response = part.text
                    elif part.function_call:
                        function_calls.append(part.function_call)

                if text_response:
                    result = {"output": text_response}
                    print(f"Respuesta del agente: {result}")
                    return result

                if function_calls:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "function_calls": function_calls
                    })

                    for function_call in function_calls:
                        try:
                            function_name = function_call.name
                            function_args = dict(function_call.args)
                                
                                # Ejecutar la función correspondiente
                                if function_name == "calcular_precio":
                                    result = calcular_precio(**function_args)
                                elif function_name == "buscar_productos":
                                    result = buscar_productos(**function_args)
                                elif function_name == "sumar_precios":
                                    result = sumar_precios(**function_args)
                                elif function_name == "verificar_descuento":
                                    result = verificar_descuento(**function_args)
                                else:
                                    result = f"Función {function_name} no implementada"
                                
                                messages.append({
                                    "role": "tool",
                                    "name": function_name,
                                    "content": result
                                })

                                print(f"Procesada llamada a función: {function_name}")
                                print(f"Resultado: {result}")
                        except Exception as e:
                            print(f"Error procesando función {function_name}: {str(e)}")
                            messages.append({
                                "role": "tool",
                                "name": function_name,
                                "content": f"Error: {str(e)}"
                            })

            except Exception as e:
                print(f"Error en llamada a Gemini: {str(e)}")
                return {"output": f"Error: {str(e)}"}
                
    except Exception as e:
        print(f"Error en target: {str(e)}")
        return {"output": f"Error procesando la solicitud: {str(e)}"}
