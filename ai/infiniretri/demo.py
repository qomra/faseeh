import time
from faseeh.ai.infiniretri.infinitri_implementation import InfiniRetri

def run_demo():
    print("Initializing InfiniRetri...")
    retriever = InfiniRetri(chunk_size=512, phrase_token_num=3, top_k=20)
    
    # Long document with multiple topics for demonstration
    long_document = """
    # Part 1: Introduction to Machine Learning

    Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data. 
    Instead of explicitly programming rules, these systems identify patterns in data and make decisions with minimal human intervention.
    The learning process begins with observations or data, such as examples, direct experience, or instruction.

    Machine learning algorithms build a model based on sample data, known as training data, to make predictions or decisions 
    without being explicitly programmed to do so. Machine learning algorithms are used in a wide variety of applications, 
    such as in medicine, email filtering, speech recognition, agriculture, and computer vision, 
    where it is difficult or unfeasible to develop conventional algorithms to perform the needed tasks.

    # Part 2: Types of Machine Learning

    There are three main types of machine learning:

    ## Supervised Learning
    Supervised learning algorithms build a mathematical model of a set of data that contains both the inputs and the desired outputs. 
    The data is known as training data, and consists of a set of training examples. Each training example has one or more inputs and 
    the desired output, also known as a supervisory signal. In supervised learning, the algorithm learns a function that maps 
    inputs to desired outputs. Common supervised learning algorithms include linear regression, logistic regression, decision trees, 
    support vector machines, and neural networks.

    The supervised learning algorithm tries to learn a function that can predict the output for new inputs. 
    It does this by analyzing the training data and producing an inferred function, which can be used for mapping new examples. 
    An optimal scenario will allow for the algorithm to correctly determine the class labels for unseen instances.

    ## Unsupervised Learning
    Unsupervised learning algorithms take a set of data that contains only inputs, and find structure in the data, 
    like grouping or clustering of data points. The algorithms, therefore, learn from test data that has not been labeled, 
    classified, or categorized. Instead of responding to feedback, unsupervised learning algorithms identify commonalities in the data 
    and react based on the presence or absence of such commonalities in each new piece of data.

    Common unsupervised learning algorithms include clustering algorithms (like k-means and hierarchical clustering), 
    principal component analysis, and autoencoders.

    ## Reinforcement Learning
    Reinforcement learning is a type of machine learning where an agent learns to behave in an environment, 
    by performing actions and seeing the results. The agent learns from the consequences of its actions, rather than 
    from being explicitly taught. It selects actions that maximize some notion of a cumulative reward.

    Reinforcement learning differs from supervised learning in that the agent is not presented with labeled examples 
    of optimal action, but must discover them through trial and error. The agent receives feedback in the form of rewards 
    or penalties as it navigates a problem space, so the algorithm is rewarded for good outcomes and penalized for bad outcomes. 
    The goal is to develop a policy that maximizes the expected cumulative reward.

    # Part 3: Deep Learning

    Deep learning is a subset of machine learning that uses neural networks with many layers (hence "deep"). 
    These deep neural networks are capable of learning from large amounts of data and can automatically discover representations 
    needed for detection or classification. They can be trained using supervised, unsupervised, or reinforcement learning.

    Deep learning has led to major breakthroughs in many fields, especially in computer vision, natural language processing, and 
    speech recognition. Convolutional Neural Networks (CNNs) have transformed computer vision tasks like image classification 
    and object detection. Recurrent Neural Networks (RNNs) and Transformer models have revolutionized natural language processing 
    and sequence modeling tasks.

    # Part 4: Machine Learning Workflow

    A typical machine learning workflow consists of:

    1. **Data Collection**: Gather the data needed for the task.
    2. **Data Preprocessing**: Clean and prepare the data for modeling.
    3. **Feature Engineering**: Create new features from the existing data that might improve model performance.
    4. **Model Selection**: Choose the appropriate model architecture for the task.
    5. **Training**: Train the model on the prepared data.
    6. **Evaluation**: Assess the model's performance on unseen data.
    7. **Tuning**: Adjust model parameters to improve performance.
    8. **Deployment**: Implement the model in a production environment.
    9. **Monitoring**: Track the model's performance over time and update as needed.

    # Part 5: Challenges in Machine Learning

    Machine learning faces several challenges:

    - **Data Quality**: Models are only as good as the data they're trained on. Poor data leads to poor models.
    - **Overfitting**: Models may perform well on training data but fail to generalize to new, unseen data.
    - **Interpretability**: Complex models like deep neural networks can be difficult to interpret and explain.
    - **Bias and Fairness**: Models can perpetuate or amplify biases present in the training data.
    - **Computational Resources**: Training complex models requires significant computational power.
    - **Privacy and Security**: Using sensitive data for training raises privacy concerns.

    # Part 6: Future of Machine Learning

    The future of machine learning looks promising with advancements in several areas:

    - **Automated Machine Learning (AutoML)**: Tools that automate the process of applying machine learning.
    - **Few-Shot and Zero-Shot Learning**: Systems that can learn from very few examples or even no examples.
    - **Explainable AI**: Making black-box models more interpretable and explainable.
    - **Federated Learning**: Training models across multiple devices while keeping data local and private.
    - **Neuromorphic Computing**: Hardware designed to mimic the structure and function of the human brain.
    - **Quantum Machine Learning**: Leveraging quantum computing for machine learning tasks.
    
    These advancements will continue to expand the capabilities and applications of machine learning in various fields.
    """
    
    # Questions to test with
    questions = [
        "What are the three main types of machine learning?",
        "What is deep learning and how does it relate to machine learning?",
        "What are the challenges faced in machine learning?",
        "What does a typical machine learning workflow consist of?",
        "What is the future of machine learning?"
    ]
    
    print("\nProcessing document...")
    start_time = time.time()
    
    # Process document once with first question (this will be our initial query)
    initial_question = questions[0]
    retriever.process_document(long_document, initial_question)
    
    processing_time = time.time() - start_time
    print(f"Document processed in {processing_time:.2f} seconds")
    
    # Test questions
    print("\n--- Testing Question Answering ---")
    for i, question in enumerate(questions):
        print(f"\nQuestion {i+1}: {question}")
        
        # For new questions, update the cache by reprocessing with the new question
        if i > 0:
            print(f"Updating cache for new question...")
            start_time = time.time()
            retriever.process_document(long_document, question)
            cache_update_time = time.time() - start_time
            print(f"Cache updated in {cache_update_time:.2f} seconds")
        
        start_time = time.time()
        answer = retriever.answer_question(question, top_k=2)
        answer_time = time.time() - start_time
        
        # Print just a preview of the answer to keep output manageable
        answer_preview = answer.split("Relevant context:")[0] + "Relevant context: [...]"
        print(f"Answer (preview): {answer_preview}")
        print(f"Retrieved in {answer_time:.2f} seconds")
    
    # Demonstrate the needle-in-haystack capability
    print("\n--- Needle in Haystack Test ---")
    # Add a very specific piece of information in a large document
    needle = "The secret code for the treasure is ALPHA9876."
    haystack = long_document + "\n\n" + "X" * 1000 + "\n\n" + needle + "\n\n" + "X" * 1000
    
    print("Processing document with hidden information...")
    needle_question = "What is the secret code for the treasure?"
    retriever.process_document(haystack, needle_question)
    
    print(f"\nQuestion: {needle_question}")
    start_time = time.time()
    answer = retriever.answer_question(needle_question, top_k=3)
    answer_time = time.time() - start_time
    print(f"Answer: {answer}")
    print(f"Retrieved in {answer_time:.2f} seconds")
    
    # Show performance comparison
    print("\n--- Performance Comparison ---")
    print("Standard search method would need to scan the entire document.")
    print(f"InfiniRetri retrieved the answer by focusing on relevant sections using attention.")
    print(f"Document size: {len(haystack)} characters")
    print(f"Cached content size: {len(retriever.tokenizer.decode(retriever.cache_tokens))} characters")
    print(f"Compression ratio: {len(retriever.tokenizer.decode(retriever.cache_tokens)) / len(haystack):.2%}")

if __name__ == "__main__":
    run_demo()