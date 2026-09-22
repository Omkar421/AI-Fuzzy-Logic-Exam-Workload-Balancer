# AI + Fuzzy Logic Exam Workload Balancer

## Abstract

Exam preparation is difficult because students must balance several uncertain factors at the same time. A subject may be difficult, the examination may be near, the student may have low confidence, and the available study time may be limited. Traditional planning tools often use fixed thresholds. For example, they may classify an examination as urgent only when fewer than seven days remain. Such thresholds do not represent gradual human judgments well.

This project presents an AI-assisted exam workload balancer. The system accepts a natural-language description of a student's situation. A LangChain pipeline extracts relevant planning signals and later explains the final recommendation conversationally. A separate fuzzy inference system processes difficulty, urgency, confidence, and study load. It uses membership functions, rule evaluation, and defuzzification to produce a workload-risk score from 0 to 100. The final Streamlit interface exposes the intermediate memberships and active rules so that the recommendation is explainable during a viva.

## 1. Problem statement and objectives

Students often make study schedules informally. They may say, “I have Operating Systems next week, it is hard, and I am not confident with memory management.” A conventional form would require the student to translate that sentence into several numeric fields before the system could respond. This creates friction and may produce incomplete inputs.

The project solves two related problems. First, it understands a natural-language study description. Second, it prioritizes subjects without pretending that workload is either urgent or not urgent. The objective is not to predict marks. The objective is to provide a pacing signal that helps the student decide what to study first.

The system has four measurable objectives. It must extract study signals from free text. It must implement genuine fuzzy reasoning. It must present an explanation of the recommendation. Finally, it must be available through a hosted user interface rather than only as a local script.

## 2. System architecture

The system contains three logical layers. The presentation layer is a Streamlit application. It accepts free text, displays extracted values, shows risk results, and reveals the reasoning trace. The AI layer contains two LangChain chains. The extraction chain converts the message into structured data. The explanation chain converts the fuzzy result into a short conversational explanation. The reasoning layer is the fuzzy engine implemented with scikit-fuzzy.

The separation between the AI and fuzzy layers is deliberate. The LLM is effective at language tasks, but its numeric output should not be trusted as the final decision mechanism. The fuzzy engine is deterministic and testable. Therefore, the LLM supplies structured inputs and natural-language output, while the fuzzy system supplies the risk score.

The main flow is:

```text
Natural-language message
        ↓
LangChain extraction chain
        ↓
Structured numeric signals
        ↓
Fuzzification
        ↓
Fuzzy rule evaluation
        ↓
Defuzzification
        ↓
Risk score and recommended focus time
        ↓
LangChain explanation chain
        ↓
Student-facing recommendation
```

## 3. AI/LLM component using LangChain

The AI component is implemented in `ai_chain.py`. It uses a `ChatPromptTemplate` to define the extraction task. The prompt instructs the model to return a JSON object containing the subject, difficulty, days left, confidence, study hours, and weekly capacity. The chain is composed with a chat model and a string output parser.

This is real language work because the user is not required to use a fixed form. The model must identify the meaning of phrases such as “in five days,” “I am weak at graphs,” or “I can study four hours daily.” The application then normalizes the extracted values before passing them to the fuzzy engine.

A second chain is used after fuzzy inference. It receives the subject, extracted inputs, risk score, and recommended focus time. It is instructed to explain the strongest signals and provide one practical action. This makes the result understandable to a student. The explanation does not replace the fuzzy result, and it is explicitly told not to claim that the score predicts marks.

A Hugging Face hosted text-generation model is used so that the application can be deployed without loading a large model into the Streamlit process. The API token is stored in Streamlit Secrets rather than committed to GitHub. During local development, the application can run with a deterministic fallback, but the submitted hosted application should be configured with a real token so that the LangChain path is active.

## 4. Fuzzy logic component

The fuzzy component uses four input variables. Difficulty is measured from 0 to 10. Days left is measured from 0 to 30 and acts as the urgency signal. Confidence is measured from 0 to 10. Study load is the estimated number of hours needed for the subject, measured from 0 to 20. The output variable is workload risk from 0 to 100.

Each variable has linguistic membership sets. Difficulty has low, medium, and high sets. Days left has low, medium, and high urgency sets, where fewer days correspond to higher urgency. Confidence has low, medium, and high sets. Study load has low, medium, and high sets. The output risk has low, medium, high, and critical sets.

The sets use triangular membership functions. A triangular function allows a value to have a partial membership between zero and one. For example, a difficulty value of six may have some membership in medium and some membership in high. This is more realistic than forcing the value into one category.

The rule base contains eight rules. The most important rules are:

1. If difficulty is high and urgency is high, then risk is critical.
2. If confidence is low and study load is high, then risk is critical.
3. If difficulty is high and confidence is low, then risk is high.
4. If urgency is high, then risk is high.
5. If study load is high and confidence is medium, then risk is high.
6. If difficulty and urgency are medium, then risk is medium.
7. If difficulty is low, confidence is high, and urgency is low, then risk is low.
8. If study load is low and confidence is high, then risk is low.

The engine first fuzzifies each crisp input. It then evaluates rule activations using fuzzy AND operations. The activated output sets are aggregated. Finally, the system uses centroid defuzzification to convert the aggregated output into one numeric risk score. This sequence satisfies the required fuzzy inference process: membership functions, fuzzification, rule evaluation, aggregation, and defuzzification.

## 5. Example and evaluation

Consider the message: “I have Data Structures in five days. It is difficulty eight, my confidence is four, I need twelve hours, and I can study twenty-four hours weekly.” The LangChain extraction chain identifies the subject and converts the values into structured signals.

The fuzzy engine sees high difficulty, high urgency, low confidence, and high study load. Multiple rules activate at the same time. The critical-risk rules receive strong activation because both difficulty and urgency are high, and because confidence is low while study load is high. The aggregated output is defuzzified into a high workload-risk score. The application recommends a larger focus block for this subject than for an easy subject with many days left.

The system can be tested with two contrasting cases. A difficult subject with five days left and low confidence should receive a high or critical score. An easy subject with twenty-five days left, high confidence, and low study load should receive a lower score. The included test file checks these broad properties and verifies that rule activations are non-zero.

The application also exposes the memberships and active rules in expandable panels. This feature is important for evaluation because it allows the student to demonstrate what happened inside the fuzzy engine rather than showing only a final number.

## 6. Limitations and future work

The model uses hand-designed membership functions and rules. These are appropriate for a mini project because they are understandable, but they are not learned from a large student dataset. The recommended study time is also a planning heuristic. It should not be treated as a guaranteed optimal schedule.

The language-extraction step can make mistakes when the student uses ambiguous wording. The application reduces this risk by showing extracted values before the final result. A future version could let the student correct the extracted values directly, store multiple subjects, and generate a calendar of study sessions.

A larger evaluation could collect anonymized examples from students and compare the fuzzy recommendations with schedules created by experienced instructors. Membership parameters could then be tuned using those examples while preserving the explainable rule base.

## Conclusion

The Exam Workload Balancer satisfies the two required technical components in separate, meaningful roles. LangChain performs natural-language understanding and conversational explanation. Fuzzy logic performs gradual, rule-based reasoning and produces the final workload score. The Streamlit interface connects these components into a hosted application that can be demonstrated during the viva. The design is intentionally explainable: every important output can be traced to an input membership and a fuzzy rule.

## References

[1]: https://docs.langchain.com/oss/python/integrations/chat/huggingface "LangChain ChatHuggingFace integration documentation"
[2]: https://scikit-fuzzy.github.io/scikit-fuzzy/ "scikit-fuzzy documentation"
[3]: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app "Streamlit Community Cloud deployment documentation"
[4]: https://docs.streamlit.io/develop/concepts/connections/secrets-management "Streamlit secrets management documentation"
