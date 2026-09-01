from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import time
import json


load_dotenv()

judge_system_prompt = """You are an expert evaluator for a RAG question-answering system.

Evaluate the generated answer using the query, reference answer, and retrieved context.

1. Answer correctness:
Determine whether the generated answer correctly answers the query and conveys the important information in the reference answer. Do not require identical wording. Give partial credit when the answer is substantially correct but incomplete or contains a minor error.

2. Faithfulness:
Determine whether the claims in the generated answer are supported by the retrieved context. Do not reward information that is merely correct in the real world if it is not supported by the context.

3. Context sufficiency:
Determine whether the retrieved context contains enough information to answer the query correctly. If the answer cannot be established from the context, score this 0.

Be strict about factual errors, unsupported claims, and contradictions, but do not penalize concise answers or different valid wording. Return only the requested structured output."""


class RAGJudge(BaseModel):
    judge_correctness: float = Field(
        description="0 = incorrect, 0.5 = partially correct, 1 = fully correct. Judge whether the generated answer correctly answers the query and agrees with the reference answer."
    )

    faithfulness: float = Field(
        description="0 = unsupported, 0.5 = partially supported, 1 = fully supported. Judge whether the claims in the generated answer are supported by the provided context."
    )

    context_sufficiency: float = Field(
        description="0 = the provided context does not contain enough information to answer the query, 1 = it contains enough information."
    )

    reason: str = Field(
        description="Brief explanation of the scores, focusing on the main error or reason."
    )


judge_llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite").with_structured_output(RAGJudge)

judge_template = ChatPromptTemplate.from_messages([
    ("system", judge_system_prompt),
    ("human", "Query: {query}\nOriginal answer: {original_answer}\nGenerated answer: {generated_answer}\nProvided context: {context}"
    )
])

judge_chain = judge_template | judge_llm

# a = judge_chain.invoke({"query":"Which game character am item?(clue: my name is jason duavl)"})
# print(a)

def get_judgments(data):
    all_evals = []
    completed_data = []

    # Resume judge-only results
    try:
        with open("evals_open_ragbench.json", "r", encoding="utf-8") as f:
            all_evals = json.load(f)
    except FileNotFoundError:
        pass

    # Resume full results
    try:
        with open(
            "evals_open_ragbench_with_queries.json",
            "r",
            encoding="utf-8"
        ) as f:
            completed_data = json.load(f)
    except FileNotFoundError:
        pass

    start_index = len(all_evals)

    print(f"Starting from query {start_index + 1}/{len(data)}")

    for idx in range(start_index, len(data)):
        item = data[idx]

        try:
            output = judge_chain.invoke({
                "query": item["query"],
                "original_answer": item["answer"],
                "generated_answer": item["generated_answer"],
                "context": item["context"]
            })

            output_dict = output.model_dump()

            # Judge-only result
            all_evals.append(output_dict)

            # Full result
            item["evals"] = output_dict
            completed_data.append(item)

            # Save judge-only checkpoint
            with open(
                "evals_open_ragbench.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    all_evals,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            # Save full checkpoint
            with open(
                "evals_open_ragbench_with_queries.json",
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    completed_data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

            print(f"Completed {idx + 1}/{len(data)}")

        except Exception as e:
            print(f"Request {idx + 1} failed: {e}")
            print("Stopping. All completed results have been saved.")
            break

        if idx < len(data) - 1:
            time.sleep(6.5)

    return completed_data


with open("rag_eval_generate_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

output = get_judgments(data)







    # query = "What is the monogenic signal and how is it used in image processing?"
    # original_answer = "The monogenic signal is a two-dimensional analog of the analytic signal, used for amplitude and phase demodulation of modulated images that are intrinsically one-dimensional. It allows for direct extraction of these features, making it a valuable tool in image processing."
    # generated_answer = "The monogenic signal (MS) is a two-dimensional analog of the analytic signal, introduced by Felsberg and Sommer [3] and independently by Larkin [7]. It allows for direct amplitude and phase demodulation of modulated images that are intrinsically one-dimensional. The MS has become a well-known tool in image processing, enabling useful steps such as equalization of brightness and phase modulation/demodulation."
    # context = "Source [1]:\nA Structurally Coherent Spatial Phase Estimate\nBrian Knight 1[0000 -0001 -8049 -4749] and Naoki Saito 1[0000 -0001 -5234 -4719]\nDepartment of Mathematics, University of California, Davis, CA 95616 USA\nAbstract. The monogenic signal (MS) was introduced by Felsberg and Sommer [3], and independently by Larkin [7] under the name vortex operator. It is a two-dimensional (2D) analog of the well-known analytic signal, and allows for direct amplitude and phase demodulation of (amplitude and phase) modulated images so long as the signal is intrinsically one-dimensional (i1D). Felsberg's PhD dissertation also introduced the structure multivector (SMV), a model allowing for intrinsically 2D (i2D) structure. While the monogenic signal has become a well-known tool in the image processing community, the SMV is little used, although even in the case of i1D signals it provides a more robust orientation estimation than the MS. We argue the SMV is more suitable in standard i1D image feature extraction due to the this improvement, and extend the steerable wavelet frames of Held et al. [4] to accommodate the additional features of the SMV. We then propose a novel quality map based on local orientation variance which values structurally coherent patches. This yields a multiscale phase estimate which performs well even when signal to noise ratio (SNR) is ≤ 1 . The performance is evaluated on several synthetic phase estimation tasks as well as on a fine-scale fingerprint registration task related to the 2D phase demodulation problem.\nKeywords: Spatial phase · Phase Demodulation · Multiscale Methods · Monogenic Signal · Fingerprint Registration\n\nSource [2]:\n2.1 The i1D Signal Model and the Monogenic Signal\nThis feature set is considered to be a split of identity , in that it separates a signal into independent local features. Specifically, the local structure is invariant to scaling of the local energy, and the local energy is invariant to phase shifts in the local structure. This allows for useful image processing steps, such as equalization of brightness [4], or, as we discuss later, phase modulation and demodulation.\nSimilarly, we can use local amplitude information in order to determine important features. This approach is particularly useful when the monogenic signal is paired with an isotropic wavelet decomposition [5], which we outline in the next section.\n\nSource [3]:\n1 Introduction\nMany problems in imaging science rely on spatial phase measurements, e.g. 2D interferometry, interferometric SAR (InSAR), and require a preprocessing step to estimate the true spatial phase of an image or set of images [12]. Any improvement to this estimate will thus improve downstream analysis. A standard approach for estimating spatial phase of images is to use the phase of the monogenic signal [6] [13]. The first improvement to this estimate is to produce a multiscale monogenic phase estimate, see Kaseb et al. [5], for instance, which makes use of isotropic wavelets [4] [10], and provides a robust phase estimate in the presence of image corruption. The main contributions of this article are: 1) to employ the structure multivector (SMV) in place of the monogenic signal in order to extract a more robust feature set at any given scale; and 2) to define a novel quality measure at each scale, based on the features of the SMV, in order to determine the optimal local feature set around a given point in an image. We perform several experiments on synthetic images to showcase the application of the our multiscale phase estimation, and further solve a phase and amplitude demodulation problem in 2D to display the utility of this estimate. Additionally, we use our multiscale phase estimate in order to solve a fine-scale fingerprint registration problem as described in [1]. Lastly, we have provided a Julia module that includes the code needed to reproduce any figures and experiments shown in this paper, as well as standalone functions to perform our multiscale phase estimation: https://gitlab.com/briancknight/SSVM2025.",
