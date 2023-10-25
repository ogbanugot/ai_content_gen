import os
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.retrievers import WikipediaRetriever

load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
standard = "K-12"
topic = "baseball"
evaluate = True  # evaluate generated content

chat = ChatOpenAI(temperature=0, openai_api_key=openai_key)
retriever = WikipediaRetriever()
docs = retriever.get_relevant_documents(query=topic)
baseball_general_knowledge = docs[0].page_content[:1227]

print("Generating content for standard: ", standard)
print("Chosen topic: ", topic)

template = (
    "You are an automated content generator for content following the {standard} standards. "
    "Your role is to ask students to "
    "read stories and literature, as well as more complex texts that provide facts and background knowledge in areas "
    "such as science and social studies. Students will be challenged and asked multiple-choice questions that push "
    "them to refer back to what they’ve read. This stresses critical thinking, problem-solving, and analytical skills "
    "that are required for success in college, career, and life. You should generate content from the provided "
    "knowledge. "
    "The generated content should be short and concise "
    "and should end with a question and multiple choice answers a to d. "
    "Make sure that the answer to the question is contained in the generated content."
    "Also, make sure that the answer to the question is contained in the multiple choice options generated."
    "Use the following examples below as a guide."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

sample1_general_knowledge = "In 1630 a group of people called Puritans left England for North America. The settlement " \
                            "they started in America was called the Massachusetts Bay Colony. The Puritans were a " \
                            "group of Protestant Christians with strict religious beliefs. They disagreed with some " \
                            "practices of England’s official church, the Church of England. The English government " \
                            "mistreated them because of their beliefs. The Puritans wanted to find a place where they " \
                            "could practice their religion in peace. In 1629 King Charles I of England gave a group of " \
                            "Puritans permission to trade and settle in America. The group was called the " \
                            "Massachusetts Bay Company. The Massachusett Bay Company sent more than 1,000 Puritans " \
                            "across the Atlantic Ocean in 1630. They were the first settlers of the Massachusetts Bay " \
                            "Colony. Their leader was John Winthrop who later became the first governor of the " \
                            "Massachusetts Bay Colony. John Winthrop gave a famous speech to Puritans who were " \
                            "traveling to North America in 1630. Winthrop described the goals of the colony that he " \
                            "planned to establish in the speech. Here is an excerpt from the speech: ‘The lord will be " \
                            "our God, and…will command a blessing upon us in all our ways…We shall find that [God] is " \
                            "among us when….men shall say of [future settlements], ‘may the lord make it like that of " \
                            "New England.’ For we must consider that we shall be as a city upon a hill. The eyes of " \
                            "all people are upon us. The Puritans set up their own government and started several " \
                            "towns. Boston was the most important town. The Puritans did not allow people who " \
                            "disagreed with their religious beliefs to live in the colony. Roger Williams was one of " \
                            "the people who was forced to leave. He then founded the colony of Rhode Island. Trouble " \
                            "gradually built up between England and the Massachusetts Bay Colony. The king had not " \
                            "wanted the colonists to govern themselves. In 1684 King Charles II put the Massachusetts " \
                            "Bay Company out of business. In 1691 England created a new Massachusetts Bay Colony by " \
                            "combining the old colony with Plymouth Colony (which had been settled by other Puritans, " \
                            "now known as the Pilgrims) and other lands. The new colony was controlled by the English " \
                            "government. "

sample1_generated_content = "One of the first permanent English settlements in North America was the Massachusetts Bay " \
                            "Colony. The leaders of the colony were Puritans who disagreed with the teachings of the " \
                            "Church of England. They wanted to reform the church, but they were persecuted for these " \
                            "views. Over the course of the 1600s, about 30,000 Puritans traveled to the English " \
                            "colonies to practice their religion freely. John Winthrop, who later became the first " \
                            "governor of the Massachusetts Bay Colony, gave a famous speech to Puritans who were " \
                            "traveling to North America in 1630. Winthrop described the goals of the colony that he " \
                            "planned to establish. Read the passage from the speech. Then answer the question below. " \
                            "‘The lord will be our God, and…will command a blessing upon us in all our ways…We shall " \
                            "find that [God] is among us when….men shall say of [future settlements], ‘may the lord " \
                            "make it like that of New England.’ For we must consider that we shall be as a city upon a " \
                            "hill. The eyes of all people are upon us. " \
                            "" \
                            "Based on the passage, what did Winthrop believe " \
                            "about the Massachusetts Bay Colony?" \
                            "a. That it would be an example for the rest of the world " \
                            "b. That it was likely to fail " \
                            "c. That it should be established on a hill " \
                            "d. That God was angry with the colony "

chain_of_thought_generation = "Provided Knowledge: " \
                              "{sample1_general_knowledge}" \
                              "" \
                              "Generated Content:" \
                              "{sample1_generated_content}" \
                              "" \
                              "Provided Knowledge:" \
                              "{topic_general_knowledge}" \
                              "" \
                              "Generated Content: "

human_message_prompt = HumanMessagePromptTemplate.from_template(chain_of_thought_generation)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

print("\nGenerating content....")
response = chat(
    chat_prompt.format_prompt(
        standard=standard,
        sample1_general_knowledge=sample1_general_knowledge,
        sample1_generated_content=sample1_generated_content,
        topic_general_knowledge=baseball_general_knowledge,
    ).to_messages()
)

baseball_generated_content = response.content

if evaluate:
    template = (
        "You are an expert QA system for generated content following {standard} standards. "
        "Your role is to evaluate the quality of the preview generated content. "
        "Use the example provided knowledge and its corresponding example generated content as a guide "
        "for the kind of quality that the preview generated content should match. "
        "You should evaluate if the preview generated content is factual, concise and informative. "
        "You should evaluate if the information in the preview generated content is actually "
        "contained in the preview provided knowledge. "
        "Score the preview generated data on a scale of 1 to 10, where 1 means the generated data is low quality "
        "and doesn't meet the criteria for factual, concise and informative "
        "and 10 means the generated data is of excellent "
        "quality and meets all the criteria. "
        "Simply provide the score between 1 and 10, no need for explanations. "
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    chain_of_thought_evaluate_gen_content = "Example Provided Knowledge: " \
                                            "{sample1_general_knowledge}" \
                                            "" \
                                            "Example Generated Content:" \
                                            "{sample1_generated_content}" \
                                            "" \
                                            "Score: 10" \
                                            "" \
                                            "Preview Provided Knowledge:" \
                                            "{topic_general_knowledge}" \
                                            "" \
                                            "Preview Generated Content:" \
                                            "{topic_generated_content}" \
                                            "" \
                                            "Score: "

    human_message_prompt = HumanMessagePromptTemplate.from_template(chain_of_thought_evaluate_gen_content)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    print("\nevaluating generated content....")
    response = chat(
        chat_prompt.format_prompt(
            standard=standard,
            sample1_general_knowledge=sample1_general_knowledge,
            sample1_generated_content=sample1_generated_content,
            topic_general_knowledge=baseball_general_knowledge,
            topic_generated_content=baseball_generated_content,
        ).to_messages()
    )
    print("Generated content score: ", response.content)

print(f"\n{baseball_generated_content}")
selected_answer = input("Please enter an option from the list of options provided above: ")

template = (
    "You are an automated evaluator of answers from students in {standard}. "
    "Your role is to evaluate the chosen option from the option of multiple choice answers."
    "You must check if the chosen option is correct according to the information in the generated content."
    "If the chosen option is correct say 'Answer <chosen option> is correct!' "
    "and if the answer is incorrect 'Answer <chosen option> is incorrect'. "
    "Of course replace <chosen option> with the actual chosen option a, b, c or d. "
    "Always provide a brief explanation for why the answer is correct or wrong. Keep it short and concise. "
    "At the end you must always provide extra information that is not in the generated content, "
    "keep it brief and informative."
    "Make sure the extra information is contained in the provided knowledge. "
    "Use the following example below as a guide."
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

chain_of_thought_evaluation = "Provided Knowledge: " \
                              "{sample1_general_knowledge}" \
                              "" \
                              "Generated Content:" \
                              "{sample1_generated_content}" \
                              "" \
                              "Selected Answer: a" \
                              "" \
                              "Evaluation of Selected Answer: Answer a is incorrect" \
                              "" \
                              "Explanation:" \
                              "Winthrop says that the Puritans want God’s blessing for the colony. " \
                              "He also says ‘The eyes of all people are upon us’. " \
                              "In other words, he believes the colony will be an example for the rest of the " \
                              "world. " \
                              "Winthrop did not mean that his settlement should actually be built on a hill. " \
                              "He used this as a metaphor for the idea that people far away would be able " \
                              "to see what the settlement did." \
                              "" \
                              "Extra Information:" \
                              "Why did the Puritans want to create a ‘city upon a hill’? In addition to " \
                              "leaving England for religious reasons, many Puritans disapproved of the " \
                              "government in England. In the Massachusetts Bay Colony, the Puritans could " \
                              "worship according to their beliefs and enforce ideal Puritan behavior." \
                              "" \
                              "Provided Knowledge:" \
                              "{topic_general_knowledge}" \
                              "" \
                              "Generated Content: {topic_generated_content}" \
                              "" \
                              "Selected Answer: {selected_answer}" \
                              "" \
                              "Evaluation of Selected Answer:" \
                              "" \
                              "Explanation:" \
                              "" \
                              "Extra Information:"

human_message_prompt = HumanMessagePromptTemplate.from_template(chain_of_thought_evaluation)

chat_prompt = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

print("\nEvaluating your answer....")
response = chat(
    chat_prompt.format_prompt(
        standard=standard,
        sample1_general_knowledge=sample1_general_knowledge,
        sample1_generated_content=sample1_generated_content,
        topic_general_knowledge=baseball_general_knowledge,
        topic_generated_content=baseball_generated_content,
        selected_answer=selected_answer,
    ).to_messages()
)
print(response.content)
