import type { TextGenerationStreamOutput } from "@huggingface/inference";
import type { Endpoint } from "../endpoints";
import { z } from "zod";

export const endpointLocalParametersSchema = z.object({
    weight: z.number().int().positive().default(1),
    model: z.any(),
    type: z.literal("local"),
    url: z.string().url().default("http://0.0.0.0:8000")
});

export function endpointLocal(
    input: z.input<typeof endpointLocalParametersSchema>
): Endpoint {
    const { url, model } = endpointLocalParametersSchema.parse(input);
    return async ({ messages, preprompt, continueMessage, generateSettings }) => {
        //console.log(preprompt);
        //console.log("messages: ", messages);
        // if preprompt starts with أنت روبوت ملخص then send the messages to the local server with /summary endpoint else send the messages to the local server with /generate_stream endpoint
        let messagesOpenAI = messages.map((message) => ({
            role: message.from,
            content: message.content,
        }));

        if (messagesOpenAI?.[0]?.role !== "system") {
            messagesOpenAI = [{ role: "system", content: "" }, ...messagesOpenAI];
        }

        if (messagesOpenAI?.[0]) {
            messagesOpenAI[0].content = preprompt ?? "";
        }
        var r = null;
        if(preprompt?.startsWith("أنت روبوت ملخص")){
            console.log("Sending messages to local server with /summary endpoint");
            r = await fetch(`${url}/summary`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    messages: messagesOpenAI,
                    stream: false
                }),
            });
            if (!r.ok) {
                throw new Error(`Failed here to generate text: ${await r.text()}`);
            }
            else{
                console.log("Success summary endpoint");
            }    
        }
        else{
            r = await fetch(`${url}/generate_stream`, {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    messages: messagesOpenAI,
                    stream: true
                }),
            });
            // get response content as json
            // const response = await r.json();
            // get generated text
            //const generatedText = response["text"];
            // console.log(generatedText);
            if (!r.ok) {
                throw new Error(`Failed here to generate text: ${await r.text()}`);
            }
            else{
                console.log("Success");
            }
        }

        const encoder = new TextDecoderStream();
        const reader = r.body?.pipeThrough(encoder).getReader();
        //console.log("encoder and reader created");

        return (async function* () {
            let generatedText = "";
            let tokenId = 0;
            let stop = false;
            let accumulatedData = ""; // Buffer to accumulate data chunks
            while (!stop) {
                // read the stream and log the outputs to console
                const out = (await reader?.read()) ?? { done: false, value: undefined };
                // we read, if it's done we cancel
                if (out.done) {
                    reader?.cancel();
                    return;
                }
                
                if (!out.value) {
                    return;
                }
                // Accumulate the data chunk
                accumulatedData += out.value;

                // Process each complete JSON object in the accumulated data
                while (accumulatedData.includes("\n")) {
                    // Assuming each JSON object ends with a newline
                    const endIndex = accumulatedData.indexOf("\n");
                    let jsonString = accumulatedData.substring(0, endIndex).trim();

                    // Remove the processed part from the buffer
                    accumulatedData = accumulatedData.substring(endIndex + 1);

                    if (jsonString.startsWith("data: ")) {
                        jsonString = jsonString.slice(6);
                        let data = null;

                        try {
                            data = JSON.parse(jsonString);   
                            //data.content = data.content.replace("<newline>", "\n");
                        } catch (e) {
                            console.error("Failed to parse JSON", e);
                            console.error("Problematic JSON string:", jsonString);
                            continue; // Skip this iteration and try the next chunk
                        }

                        // Handle the parsed data
                        if (data.content || data.stop) {
                            generatedText += data.content;
                            // replace word <newline> with \n
                            
                            const output: TextGenerationStreamOutput = {
                                token: {
                                    id: tokenId++,
                                    text: data.content ?? "",
                                    logprob: 0,
                                    special: false,
                                },
                                generated_text: data.stop ? generatedText : null,
                                details: null,
                            };
                            if (data.stop) {
                                console.log("data stop: ", data.stop);
                                stop = true;
                                output.token.special = true;
                                reader?.cancel();
                            }
                            yield output;
                        }
                    }
                }
            }
        })();
    };
}

export default endpointLocal;