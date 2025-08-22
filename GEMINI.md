# Project Context

## 1. Project Goal
Implement and reproduce various model extraction attacks.

## 2. Primary Objective
Implement attack techniques as concretely as possible based on the proposed papers to reproduce results.

## 3. Implementation Philosophy
Attack techniques should be implemented meticulously and concretely, while other elements should be implemented as simply, reusably, and intuitively as possible.

## 4. Data and Specific Tasks
The implementation will be based on:
*   MNIST dataset
*   CIFAR10 dataset
*   User's existing "sewer pipe defect detection binary classification model" and associated personal data.

## 5. Reference
Model extraction attack reference paper
*   tramer 
    *   paper: https://www.usenix.org/system/files/conference/usenixsecurity16/sec16_paper_tramer.pdf
    *   github: https://github.com/ftramer/Steal-ML
*   knockoff nets
    *   paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Orekondy_Knockoff_Nets_Stealing_Functionality_of_Black-Box_Models_CVPR_2019_paper.pdf
    *   github: https://github.com/tribhuvanesh/knockoffnets
*   papernot
    *   paper: https://www.cs.purdue.edu/homes/bb/2020-fall-cs590bb/docs/at/attacks-against-machine-learning.pdf
    *   github: None    
*   prada
    *   paper: https://ar5iv.labs.arxiv.org/html/1805.02628
    *   github: https://github.com/SSGAalto/prada-protecting-against-dnn-model-stealing-attacks
*   dfme
    *   paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Truong_Data-Free_Model_Extraction_CVPR_2021_paper.pdf
    *   github: https://github.com/cake-lab/datafree-model-extraction
*   cloud leak
    *   paper: https://www.ndss-symposium.org/wp-content/uploads/2020/02/24178.pdf
    *   github: https://github.com/yunyuntsai/DNN-Model-Stealing
*   maze
    *   paper: https://openaccess.thecvf.com/content/CVPR2021/papers/Kariyappa_MAZE_Data-Free_Model_Stealing_Attack_Using_Zeroth-Order_Gradient_Estimation_CVPR_2021_paper.pdf
    *   github: https://github.com/sanjaykariyappa/MAZE

## 6. Current Progress

### Refactoring and Implementation:
*   **Project Structure:** Refactored into a modular Python package (`model_extraction_attack`).
*   **Configuration:** Switched to a YAML-based configuration system.
*   **Attack Implementations:**
    *   `Papernot Attack`: Implemented faithfully to the paper's core methodology.
    *   `Knockoff Attack`: Implemented faithfully to the paper's core methodology, including adaptive querying and reward functions.
    *   `Tramer Attack`: Implemented faithfully to the paper's core strategies (uniform, adaptive, linesearch).
    *   `Prada Attack`: Implemented faithfully, orchestrating sub-attacks and reusing crafter methods.
    *   `Dfme Attack`: Implemented as a simplified placeholder due to complexities in wrapping the external repository.
    *   `CloudLeak Attack`: Implemented as a simplified placeholder due to complexities in wrapping the external repository.
    *   `Maze Attack`: Implemented as a simplified placeholder due to complexities in wrapping the external repository.
*   **Crafter Module:** Created `crafter.py` to house general synthetic sample generation methods, making them reusable.
*   **Metrics:** Integrated `calculate_papernot_transferability` and `agreement` metrics.

### Testing and Documentation:
*   **Testing:** Added basic `pytest` tests for all implemented attacks (both faithful and simplified versions).
*   **Documentation:** Added comprehensive docstrings to all major classes and functions for improved readability and maintainability.

### External Repository Integration Attempt (DFME):
*   Attempted to wrap the external DFME repository for faithful reproduction.
*   Encountered significant challenges with external dependencies, model loading compatibility, and unclear saving mechanisms within the external repository's code.
*   Decision was made to revert to simplified implementations for DFME, CloudLeak, and Maze to ensure overall project completion and maintainability, as per user's revised instruction.
