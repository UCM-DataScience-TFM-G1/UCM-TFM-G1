import os
import sys
sys.path.append('../..')

import panel as pn  # GUI
pn.extension()

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
import pandas as pd
#langchain
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import Runnable
from langchain.schema.runnable.config import RunnableConfig
#chainlit
import chainlit as cl

@cl.on_chat_start
async def on_chat_start():
        # load documents
    train_path = r"C:\Users\alvar\Desktop\UCM-TFM-G1\data\LLM\waste-classification\train.csv"
    test_path = r"C:\Users\alvar\Desktop\UCM-TFM-G1\data\LLM\waste-classification\test.csv"
    validation_path = r"C:\Users\alvar\Desktop\UCM-TFM-G1\data\LLM\waste-classification\validation.csv"

    # Read the CSV files into DataFrames with the first column as the index
    train_df = pd.read_csv(train_path, index_col=0)
    test_df = pd.read_csv(test_path, index_col=0)
    validation_df = pd.read_csv(validation_path, index_col=0)

    # Concatenate DataFrames vertically
    merged_df = pd.concat([train_df, test_df, validation_df], ignore_index=True)
    # Rename the 'Phrase' column to 'Question'
    merged_df.rename(columns={'Phrase': 'Question'}, inplace=True)

    # Create a new column 'Answer'
    merged_df['Answer'] = ''
    # Define answers for each waste class

    answer_mapping = {
        
        "TOBACCO PACK": "Tobacco packs, often made of a mix of materials like paper, plastic, and foil, pose a challenge in recycling due to the combination. It's recommended to dispose of tobacco packs in the general waste bin. To reduce environmental impact, consider minimizing tobacco product use or exploring alternative, more sustainable packaging options.",
        
        "CONDIMENT PACKETS": "Condiment packets, such as ketchup or soy sauce packets, vary in recyclability based on material. Single-material packets like paper are usually recyclable, while multi-material packets may not be. Check local recycling guidelines and, if possible, separate single-material packets for recycling. Minimize use of single-use packets by opting for larger containers when available.",

        "TRANSPORT TICKET": "Transport tickets, commonly made of paper, are generally recyclable. To ensure proper recycling, place used tickets in the paper recycling bin. However, if the ticket has a magnetic strip or other non-paper components, it may not be recyclable. Always check local recycling instructions for specific guidelines on ticket disposal. Consider digital tickets as an eco-friendly alternative.",

        "PAPER CUP": "Paper cups, commonly used for coffee and beverages, may or may not be recyclable depending on the presence of a plastic lining. Unfortunately, the plastic lining in many paper cups makes them non-recyclable in standard recycling systems. To be more sustainable, use reusable cups or consider bringing your own cup to reduce single-use waste.",

        "GLASS BOTTLE": "Glass bottles are highly recyclable and can be recycled indefinitely without losing quality. To recycle glass bottles, place them in the designated glass recycling bin. Recycling glass helps conserve resources and reduces energy consumption compared to producing new glass. Always remember to remove any caps or lids before recycling.",

        "ALUMINIUM CAN": "Aluminium cans are highly recyclable and valuable in recycling systems. To recycle aluminium cans, simply place them in the designated recycling bin for metal. Recycling aluminium helps conserve natural resources and reduces energy consumption. Ensure cans are clean before recycling for better results.",

        "ORGANIC SCRAPS": "Organic scraps, such as fruit and vegetable peels, coffee grounds, and food scraps, are excellent candidates for composting. Create a composting system at home or check with local composting facilities. Composting organic waste reduces landfill contributions and produces nutrient-rich soil for gardening.",

        "COMPOSTABLE PACKAGING": "Compostable packaging, typically made from plant-based materials, can be composted in industrial composting facilities. Avoid mixing compostable items with regular recycling, as they require specific conditions for decomposition. Check local composting guidelines or facilities that accept compostable materials.",

        "PAPER MAGASINE": "Paper magazines are recyclable and can be included in the paper recycling bin. Ensure that magazines are clean and free from contaminants for optimal recycling. Consider donating or sharing magazines with others to extend their usability before recycling.",

        "CIGARETTE PACK": "Cigarette packs, often made of paper and foil, are not easily recyclable due to contamination from tobacco residue. It's recommended to dispose of cigarette packs in the general waste bin. Encourage responsible disposal to minimize environmental impact.",

        "PAPER SHEET": "Paper sheets, whether used for notes or printing, are generally recyclable. Place clean paper sheets in the paper recycling bin. If paper sheets have confidential information, consider shredding before recycling. Reducing paper use and opting for digital alternatives are sustainable practices.",

        "PHONE CHARGER": "Phone chargers, typically composed of plastic and metal components, can be recycled. Check for designated electronic waste (e-waste) recycling facilities in your area. If the charger is still functional, consider donating or recycling through electronic recycling programs to reduce electronic waste.",

        "PAPER BOWL": "Paper bowls, commonly used for food, may or may not be recyclable based on their coating. If the bowl has a plastic lining, it may not be recyclable in standard systems. Consider using alternatives like reusable bowls or those with compostable coatings. Always check local recycling guidelines for specific instructions.",

        "PLASTIC CAP": "Plastic caps, often found on bottles, can be recycled if they are made from the same type of plastic as the bottle. Check local recycling guidelines for plastic types accepted in your area. If unsure, it's better to remove the cap and place it in the general waste bin to avoid contamination.",

        "CYLINDRICAL BATTERY": "Cylindrical batteries, commonly used in various devices, should be disposed of as hazardous waste. Many communities have specific collection points or programs for battery recycling. Avoid disposing of batteries in regular waste to prevent environmental harm. Consider using rechargeable batteries for a more sustainable option.",
        
        "MEDS BLISTER": "Medication blisters, commonly used for packaging pills, are challenging to recycle due to their mixed material composition. Dispose of meds blister packaging in the general waste bin. Consider participating in medication take-back programs to ensure proper disposal and prevent environmental harm from pharmaceuticals.",

        "PLASTIC BOTTLE": "Plastic bottles, made from PET or other recyclable plastics, are highly recyclable. Empty and clean plastic bottles before placing them in the recycling bin. Recycling plastic bottles conserves resources, reduces landfill waste, and supports the production of new plastic products. Check local recycling guidelines for specific plastic types accepted.",

        "SMARTPHONE": "Smartphones, electronic devices with various components, should be recycled through designated electronic waste (e-waste) recycling programs. Many manufacturers and retailers offer smartphone recycling options. Ensure data is securely erased before disposal. Consider donating or selling functional devices to extend their lifespan.",

        "LAPTOP CHARGER": "Laptop chargers, consisting of plastic and metal components, can be recycled through electronic waste (e-waste) recycling programs. Avoid disposing of chargers in regular waste to prevent environmental harm. Check for collection points or programs that accept electronic accessories in your area.",

        "TEA BAG": "Tea bags, often made of a combination of paper and plastic, are not universally compostable. Check for compostable tea bag options, and if compostable, dispose of them in compost bins. Otherwise, discard used tea bags in the general waste bin. Consider loose-leaf tea or compostable alternatives for a more sustainable choice.",

        "PAPER PACKAGING": "Paper packaging, such as boxes and cartons, is generally recyclable. Flatten and clean paper packaging before placing it in the paper recycling bin. Recycling paper packaging helps reduce the demand for new paper production and minimizes environmental impact. Remove any non-paper components if necessary.",

        "PLASTIC BAG": "Plastic bags, commonly used for shopping, are often not accepted in regular recycling bins due to their lightweight nature. Many grocery stores have plastic bag recycling bins. Alternatively, reuse plastic bags or bring reusable bags to reduce single-use plastic consumption. Dispose of plastic bags responsibly to prevent litter.",

        "PAPER TRAY": "Paper trays, commonly used for food packaging, may be recyclable depending on their coating. Check local recycling guidelines for paper types accepted in your area. If the tray is coated with plastic or other materials, it may not be recyclable. Consider alternatives like reusable or compostable trays for sustainable choices.",

        "PLASTIC DISH": "Plastic dishes, including plates and utensils, may not be universally recyclable due to variations in plastic types. Check local recycling guidelines for accepted plastic types. Consider using reusable or biodegradable alternatives to reduce single-use plastic waste. Proper disposal in the general waste bin is advised if recycling is not an option.",

        "PLASTIC TRAY": "Plastic trays, often used for packaging food items, may or may not be recyclable depending on the plastic type. Check local recycling guidelines for specific instructions. If the tray is contaminated with food or made from non-recyclable plastics, dispose of it in the general waste bin. Minimize single-use plastic consumption where possible.",

        "PLASTIC CUP": "Plastic cups, commonly used for beverages, may be recyclable depending on the plastic type. Check local recycling guidelines for accepted plastics. Clean plastic cups before recycling. If the cup is non-recyclable or contaminated, dispose of it in the general waste bin. Consider using reusable cups to reduce single-use plastic waste.",

        "GLASS JAR": "Glass jars are highly recyclable and can be placed in the glass recycling bin. Ensure jars are clean and free from contaminants for optimal recycling. Recycling glass reduces the demand for new glass production and conserves energy. Consider repurposing glass jars for storage or donating them for reuse.",

        "PLASTIC PACKAGING": "Plastic packaging comes in various forms, from containers to wraps, and its recyclability depends on the specific type of plastic. Check local recycling guidelines for accepted plastic types. Empty and clean plastic packaging before recycling to ensure contamination-free processing. Reducing reliance on single-use plastics and choosing products with minimal packaging can contribute to waste reduction.",

        "PLASTIC GLOVES": "Disposable plastic gloves, often used for hygiene purposes, are not recyclable and should be disposed of in the general waste bin. Properly discard used gloves after single use to prevent the spread of germs. Consider using reusable or alternative materials for tasks that don't require single-use gloves.",

        "MIXED PAPER-PLASTIC PACKAGING": "Mixed paper-plastic packaging, combining paper and plastic elements, can be challenging to recycle due to the different materials. Check local recycling guidelines for specific instructions. If separation is required, carefully disassemble components. Consider supporting products with easily recyclable packaging to promote waste reduction.",

        "PLASTIC SNACK PACKAGING": "Plastic snack packaging, commonly used for chips and snacks, may not be universally recyclable due to its composition. Check local recycling guidelines for specific instructions. Empty and clean snack packaging before recycling. Consider choosing snacks with minimal packaging or in recyclable materials for more sustainable options.",

        "FACE MASK": "Face masks, used for protection, are considered single-use items and should be disposed of in the general waste bin. Avoid littering and follow proper disposal practices. Consider using reusable cloth masks to minimize environmental impact and ensure proper waste management.",

        "METAL CAP": "Metal caps, often found on glass or plastic bottles, are recyclable. Remove metal caps and lids from containers before recycling. Recycling metal conserves resources and energy. Ensure metal caps are clean and free from contaminants for optimal recycling. Check local guidelines for metal recycling.",

        "PAPER FOOD PACKAGING": "Paper food packaging, such as takeout containers or wrappers, is generally recyclable. Clean and flatten paper food packaging before recycling to minimize contamination. Recycling paper supports sustainable practices by reducing the demand for new paper production. Remove any non-paper components if necessary.",

        "CRUMBLED TISSUE": "Crumbled tissue, used tissue paper or napkins, is not recyclable due to its low-quality fiber. Dispose of crumbled tissue in the general waste bin. Consider composting tissue made from natural fibers in a home composting system. Use tissue sparingly to reduce overall waste.",

        "PLASTIC CUTLERY": "Plastic cutlery, including forks, knives, and spoons, may not be universally recyclable. Check local recycling guidelines for accepted plastics. If not recyclable, dispose of plastic cutlery in the general waste bin. Consider using reusable or compostable alternatives to reduce single-use plastic waste.",

        "RECEIPT": "Receipts, often made of thermal paper, are not universally recyclable due to the presence of chemicals. Dispose of receipts in the general waste bin. Consider opting for electronic receipts or paperless alternatives to reduce paper waste and promote sustainability.",

        "WOODEN STICKS": "Wooden sticks, such as popsicle sticks or coffee stirrers, can be composted if made from natural, untreated wood. Dispose of wooden sticks in the compost bin if available. Avoid contaminating compost with non-compostable materials. Choose products with wooden sticks over plastic alternatives for eco-friendly choices.",

        "WOODEN CUTLERY": "Wooden cutlery, like forks and spoons, is generally compostable if made from untreated wood. Dispose of wooden cutlery in the compost bin, ensuring it's free from contaminants. Opt for wooden cutlery over plastic options for a more sustainable and eco-friendly choice.",

        "CIGARETTE BUTT": "Cigarette butts are not recyclable and should be disposed of in the general waste bin. Cigarette filters are made of non-biodegradable materials and can contribute to environmental pollution. Encourage responsible disposal practices and consider using designated receptacles for cigarette waste.",

        "PLASTIC BOWL": "Plastic bowls, commonly used for serving food, may or may not be recyclable depending on the plastic type. Check local recycling guidelines for accepted plastics. Clean plastic bowls before recycling. If the bowl is non-recyclable or contaminated, dispose of it in the general waste bin. Consider using reusable or compostable bowls for more sustainable choices.",

        "ALUMINIUM SHEET": "Aluminium sheets are recyclable and can be included in aluminum recycling programs. Ensure that the sheets are clean and free from contaminants before recycling. Recycling aluminum saves energy and resources, contributing to environmental sustainability. Check local guidelines for specific instructions on recycling aluminum sheets.",

        "PAPER PLATE": "Paper plates are generally recyclable if they are not contaminated with food or other substances. Dispose of clean paper plates in the recycling bin. Recycling paper plates reduces the demand for new paper production and promotes sustainable waste management practices. Avoid coating paper plates with wax or plastic if recyclability is a concern.",

        "TETRAPACK": "Tetra Paks, commonly used for packaging liquids like milk and juice, are recyclable. Check local recycling guidelines for specific instructions. Rinse and flatten Tetra Paks before recycling to minimize contamination. Recycling Tetra Paks conserves resources and reduces environmental impact. Ensure proper disposal in accordance with local regulations.",

        "PLASTIC STICKS": "Plastic sticks, such as cotton swab sticks or stirrers, may not be universally recyclable. Check local recycling guidelines for accepted plastics. If not recyclable, dispose of plastic sticks in the general waste bin. Consider using alternatives like paper or reusable materials to reduce single-use plastic waste. Proper disposal contributes to waste reduction and environmental preservation.",

        "PLASTIC STRAW": "Plastic straws are often not recyclable due to their small size and composition. Dispose of plastic straws in the general waste bin, as they may contribute to environmental pollution. Consider using reusable or biodegradable straw alternatives to minimize single-use plastic waste. Encourage responsible disposal practices to protect the environment.",

        "PIZZA BOX": "Pizza boxes are recyclable if they are not heavily soiled with grease or food residue. Remove any remaining pizza or food scraps, and tear off any clean portions for recycling. Contaminated or greasy parts should be discarded in the general waste bin. Recycling pizza boxes supports waste reduction efforts and promotes environmental sustainability.",

        "COVID TEST": "COVID tests, often made with a combination of materials, may have specific disposal guidelines. Follow local health and safety regulations for the disposal of COVID test materials. If possible, separate components based on recyclability and dispose of each accordingly. Prioritize proper disposal methods to ensure safety and environmental responsibility.",

        "PAPER SUGAR BAG": "Paper sugar bags are generally recyclable, provided they are free from contaminants. Dispose of clean paper sugar bags in the recycling bin to support sustainable waste management practices. Recycling paper bags reduces the demand for new paper production and minimizes environmental impact. Check local guidelines for specific instructions on paper recycling.",

        "ALUMINIUM TRAY": "Aluminium trays, often used for packaging or cooking, are recyclable. Ensure that aluminium trays are clean and free from contaminants before recycling. Recycling aluminium conserves resources and reduces energy consumption. Proper disposal in aluminium recycling programs contributes to waste reduction and environmental sustainability."
    }

    # Update the 'Answer' column based on the mapping
    merged_df['Answer'] = merged_df['Class'].map(answer_mapping)
    # Assuming 'Class_index' is the column you want to remove
    merged_df = merged_df.drop(columns=['Class_index'])
    test_df = merged_df.head(1000)
    loader = DataFrameLoader(test_df, page_content_column="Answer")
    data=loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(data)
    # define embedding
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-l6-v2')
    # create vector database from data
    db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    template = """Your name is AngryGreta and you are a recycling chatbot. Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. Use three sentences maximum. Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
    Helpful Answer:"""
    prompt = PromptTemplate(input_variables=["context", "question"],template=template,)
    llm = HuggingFaceHub(repo_id = "google/flan-t5-xxl", model_kwargs={"max_length": 512, "temperature": 0.5})
    question_generator_chain = LLMChain(llm=llm, prompt=prompt)
    chain = ConversationalRetrievalChain(
    combine_docs_chain=combine_docs_chain,
    retriever=retriever,
    question_generator=question_generator_chain,
    )
    cl.user_session.set("chain", chain)

@cl.on_message
async def on_message(message: cl.Message):
    chain = cl.user_session.get("chain")  # type: Runnable

    msg = cl.Message(content="")

    async for chunk in runnable.astream(
        {"question": message.content},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await msg.stream_token(chunk)

    await msg.send()