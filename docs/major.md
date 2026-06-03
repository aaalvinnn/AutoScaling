RE: TCC-2025-08-0422, "AutoLFD: A Three-Stage Framework for Microservice Fine-grained Auto-scaling in Edge Server Cluster"
Manuscript Type: Regular paper

Dear Dr. Zhang,

We have completed the review process of the above referenced paper for the Transactions on Cloud Computing and recommend that your paper undergo a Major Revision.  

Enclosed are your reviews. If you should choose to revise your paper, please prepare a separate document describing how each of the reviewers' comments are responded to in your revision and submit by 02-Jul-2026.

Once the revised manuscript is prepared, you can upload it and submit it through your Author Center.

When you are ready to submit your revision, visit the following link:
https://ieee.atyponrex.com/submission/submissionBoard/REX-PROD-2-3F7B3365-233C-4B37-9E8B-42A862508369-340BAEE0-95A0-4304-8533-11C16F7589EE-43622/current?idtype=external

The revised manuscript must contain the following:

-abstract
-index terms
-affiliation information
-main text
-references
-figure captions
-table titles
-brief biographies of each author

If you have appendices or supplementary material, it must be submitted as a separate document.

When submitting your revised manuscript, you will be able to respond to the comments made by the reviewer(s) in the space provided. You can use this space to document any changes you make to the original manuscript. In order to expedite the processing of the revised manuscript, please be as specific as possible in your response to the reviewer(s)’ questions and comments. You may also upload your responses as separate files for review along with your revision. If you choose to do this, please choose “Summary of Changes” as the file designation. You may not designate the changes in the text by using colored, bold, or italic text.

Please note, some reviewers may have recommended that you discuss additional literature when revising your manuscript. If you feel that the recommended literature does not contribute to the scholarly content of the article or is otherwise irrelevant, please note your concerns in your response to reviewer feedback.

When the submission process is complete, you will receive an automated confirmation email immediately. If you did not receive that email, your submission is not yet complete.  

The Administrator will contact you should we have any concerns or questions regarding your revision. Otherwise, your revision will be forwarded to the assigned Associate Editor with a request to begin the second round of reviews.

Please be mindful when making your revisions that you still need to maintain the size limitations for papers submitted to Transactions on Cloud Computing. Our manuscript types and submission length guidelines (including the main text, the abstract, index terms, illustrations and references) are as follows:

Transactions on Cloud Computing manuscript types and submission length guidelines are found at,

http://www.computer.org/portal/web/peerreviewjournals/author#manuscript

Please note that double column will translate more readily into the final publication format.  Our peer review double column templates can be found at,

http://www.computer.org/portal/web/peerreviewjournals/author#templates

Please do not hesitate in contacting me should you have any questions about our process or are experiencing technical difficulties. Thank you for your contribution to Transactions on Cloud Computing, and we look forward to receiving your revised manuscript.

Sincerely,

Quan Chen, AEIC
Transactions on Cloud Computing
chen-quan@sjtu.edu.cn, chen-quan@cs.sjtu.edu.cn

**************

Associate Editor
Comments to the Author:
The authors should address the concerns raised by the reviewers and submit the revised manuscript for a second round of reviews.

********************

Reviewer Comments

Reviewer: 1

Recommendation: Author Should Prepare A Major Revision

Comments:
1.The proposed MDRL algorithm demonstrates excellent performance in the small-scale experimental environment. However, its flat action space (server, microservice, change_amount) may face significant scalability challenges in large-scale clusters.
2. Regarding the historical request data Rhis in the state representation, the paper should clarify how the DRL network utilizes this time-series information for implicit prediction.
3. I suggest adding an ablation study to illustrate a direct quantification of the specific contribution of Lyapunov optimization within the framework. For instance, a variant of the algorithm could be designed where the DRL reward function does not incorporate the Lyapunov term but instead uses a more intuitive reward composed of a weighted sum of latency and cost.

Additional Questions:
1. Which category describes this manuscript?: Research

2. How relevant is this manuscript to the readers? Explain under Public Comments.: Relevant

1. Please explain how this manuscript advances the field of research and/or contributes something new to the literature.: This paper presents an innovative three-stage framework, AutoLFD, which effectively addresses the dynamic microservice auto-scaling problem in edge environments by integrating Lyapunov optimization with deep reinforcement learning. The use of the Lyapunov drift-plus-penalty term directly as the DRL reward signal is a highly novel and theoretically sound design that deserves significant credit.

2.  Is the manuscript technically sound? Please explain your answer under Public Comments below.: Appears to be - but didn't check completely

1. Are the title, abstract, and keywords appropriate? Please explain under Public Comments below.: Yes

2. Does the manuscript contain sufficient and appropriate references? Please explain under Public Comments below.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.


If you are not suggesting any references, please type NA.: N/A

3. Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? Please explain under Public Comments below.: Could be improved

4. How would you rate the organization of the manuscript? (Is it focused? Is the length appropriate for the topic?) Please explain under Public Comments below.: Could be improved

5. Please rate the readability of the manuscript. Please explain under Public Comments below.: Easy to read

6. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Does not apply, no supplementary files included

7. If yes to 6, should it be accepted:

Please rate the manuscript. Explain your choice: Excellent


Reviewer: 2

Recommendation: Author Should Prepare A Major Revision

Comments:
Summary
This paper proposes AutoLFD, a three-stage framework for fine-grained auto-scaling of microservices in edge server clusters. The algorithm integrates Lyapunov optimization, First-Fit Diminishing (FFD), and Deep Reinforcement Learning (DRL) to jointly optimize microservice deployment and request routing. By modeling the system with an open Jackson queuing network, the authors transform the long-term optimization of latency and cost into a tractable per-slot problem. Extensive experiments demonstrate that AutoLFD outperforms state-of-the-art algorithms, reducing latency by 9.86% and cost by 8.02%. The paper provides a well-structured theoretical foundation and presents a comprehensive algorithmic design supported by quantitative evaluations.

Comments for the authors:
1.The paper lacks a comprehensive comparison with state-of-the-art methods that also consider joint optimization of deployment and routing. While the proposed AutoLFD is compared against PPA, ProScale, and a basic RL agent, it would be beneficial to include more recent works that explicitly model the coupling between microservice deployment and request routing, such as those using graph neural networks or advanced multi-agent reinforcement learning.
2.The communication latency model is oversimplified. The assumption of identical and constant communication latency T0 between all servers may not hold in real edge environments with heterogeneous network conditions. A more realistic model incorporating variable bandwidth, congestion, or wireless channel effects would strengthen the practical applicability of the framework.
3.The DRL-based scaling strategy relies heavily on accurate request arrival rate prediction, but the prediction mechanism is not thoroughly evaluated. The paper uses historical data for prediction, but no details are provided on the prediction model, its accuracy, or how prediction errors impact the scaling performance. Sensitivity analysis under prediction noise would be valuable.
4.The experimental evaluation is limited to synthetic and one real-world trace (Twitter). Broader validation using multiple real-world microservice traces (e.g., from Alibaba Cluster Trace or Google Cluster Data) would better demonstrate the generalizability of AutoLFD across different workload patterns.
5.The paper does not discuss the training overhead or convergence time of the MDRL module. In real-time edge scenarios, the computational cost and time required for DRL training and inference are critical and should be quantified to assess deployability.
6.The joint optimization of deployment and routing is decoupled into separate stages (MFFD + MDRL), which may lead to suboptimal solutions. An end-to-end trainable architecture that jointly optimizes both aspects in a single DRL framework could be explored to avoid potential performance gaps.

Additional Questions:
1. Which category describes this manuscript?: Research

2. How relevant is this manuscript to the readers? Explain under Public Comments.: Relevant

1. Please explain how this manuscript advances the field of research and/or contributes something new to the literature.: This manuscript advances the field by proposing a unified framework (AutoLFD) for fine-grained microservice auto-scaling in edge server clusters. Unlike prior works that treat deployment and routing separately, AutoLFD jointly optimizes both through a three-stage approach integrating Lyapunov optimization, heuristic FFD deployment, and DRL-based dynamic scaling. The framework effectively balances latency and cost under dynamic workloads, demonstrating notable improvements over existing baselines. The combination of theoretical modeling and data-driven learning represents a meaningful step toward adaptive, intelligent resource management in edge–cloud systems.

2.  Is the manuscript technically sound? Please explain your answer under Public Comments below.: Appears to be - but didn't check completely

1. Are the title, abstract, and keywords appropriate? Please explain under Public Comments below.: Yes

2. Does the manuscript contain sufficient and appropriate references? Please explain under Public Comments below.: References are sufficient and appropriate

If you are suggesting additional references they must be entered in the text box provided.  All suggestions must include full bibliographic information plus a DOI.


If you are not suggesting any references, please type NA.: NA

3. Does the introduction state the objectives of the manuscript in terms that encourage the reader to read on? Please explain under Public Comments below.: Yes

4. How would you rate the organization of the manuscript? (Is it focused? Is the length appropriate for the topic?) Please explain under Public Comments below.: Could be improved

5. Please rate the readability of the manuscript. Please explain under Public Comments below.: Readable - but requires some effort to understand

6. Should the supplemental material be included? (Click on the Supplementary Files icon to view files): Yes, as part of the main paper if accepted (cannot exceed the strict page limit)

7. If yes to 6, should it be accepted: As is

Please rate the manuscript. Explain your choice: Excellent