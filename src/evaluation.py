from google import genai
from google.genai import types
from langsmith import Client, traceable
from langsmith.schemas import Run, Example
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

load_dotenv()

gemini_client = genai.Client()
client = Client()

# Función objetivo que simula el comportamiento del agente en la vida real.
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


# Evaluador de amabilidad donde se evalúa si el agente fue amable al responder en la vida real.
def kindness(run: Run, example: Example) -> dict:
    # Extraer datos del run y example
    response_text = run.outputs.get("output", "") if run.outputs else ""
    question = example.inputs.get("question", "") if example.inputs else ""
    expected = example.outputs.get("answer", "") if example.outputs else ""

    system_instruction = """
                Evaluar la respuesta del agente:

                ¿El agente fue amable al responder en la vida real? Independientemente de si encontró o no el producto, o del costo del producto.

                - True: Si fue amable
                - False: No fue amable

                Algunos factores que puedes tomar en cuenta y no son excluyentes son: efusividad, voluntad de ayudar y de presentar el producto con buenas referencias
                """

    user_content = f"""
                    Pregunta del cliente fue: {question}
                    La respuesta de referencia esperada fue: {expected}
                    La respuesta real del agente fue: {response_text}
                """

    config = types.GenerateContentConfig(
        system_instruction=system_instruction
    )

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=user_content,
        config=config
    )

    result = {
        "key": "kindness",
        "score": response.candidates[0].content.parts[0].text.strip().lower() == "true"
    }

    return result


# Evaluador de emojis donde se evalúa si el agente coloca emojis en su respuesta.
def contains_emoji(run: Run, example: Example) -> dict:
    common_emojis = [
        "😊", "❤️", "👍", "😂", "🙌", "✨", "🎉", "🔥", "💪", "👏",
        "🌟", "💯", "🤔", "👀", "💜", "✅", "🎈", "🌈", "🙏", "⭐",
        "💻", "📱", "🖥️", "⌨️", "🖱️", "💾", "📦", "🛒", "🛍️", "🔋",
        "🧑‍💻", "📡", "📊", "📈", "🖋️", "🖇️", "🏷️", "💳", "💡", "🔧"
    ]
    response_text = run.outputs.get("output", "") if run.outputs else ""
    has_emoji = any(emoji in response_text for emoji in common_emojis)
    result = {
        "key": "contains_emoji",
        "score": has_emoji
    }
    return result


# Nombre del dataset a evaluar
dataset_name = "Platzi Store Dataset v2"

# Ejecuta la evaluación del agente
experiment = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[kindness, contains_emoji],
    experiment_prefix="platzi-store-eval",
    description="Mide la amabilidad del agente y si coloca emojis en su respuesta",
    max_concurrency=1
)
