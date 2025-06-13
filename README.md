# **QUBO Model: Optimized Warehouse-to-Customer Assignment**

## Contents of `QUBO_model` Folder

### **1. Overview**
This Quadratic Unconstrained Binary Optimization (QUBO) model is designed to optimize **warehouse-to-customer assignments** by minimizing transportation costs while ensuring that each customer is assigned to exactly one warehouse. The model incorporates **biasing** to prefer certain warehouse assignments and visualizes the optimized solution in a **distance-scaled network graph**.

---

### **2. Objective Function: Minimize Transportation Cost**
The primary objective of the model is to **minimize the total transport cost** based on the **distances between warehouses and customers**. This is achieved using the following cost function:

$$
\text{Cost} = \sum_{w=0}^{W-1} \sum_{c=0}^{C-1} d_{w,c} \cdot y_{w,c}
$$

Where:
- $ W $ = total number of warehouses  
- $ C $ = total number of customers  
- $ d_{w,c} $ = distance between warehouse $ w $ and customer $ c $  
- $ y_{w,c} $ is a **binary variable** that equals **1** if warehouse $ w $ serves customer $ c $, otherwise it is **0**.

The **solver aims to minimize this function**, which means it finds assignments that reduce the overall transportation cost.

---

### **3. Constraints: Each Customer Must Be Assigned to Exactly One Warehouse**
A key requirement in the model is that **each customer must be assigned to exactly one warehouse**. This is formulated as:

$$
\sum_{w=0}^{W-1} y_{w,c} = 1, \quad \forall c \in C
$$

This means that for each customer $ c $, the sum of all warehouse assignments must equal **1** (the customer cannot be left unassigned or assigned to multiple warehouses).  

To enforce this constraint in the QUBO model, we use a **quadratic penalty function**:

$$
P \cdot \left(\sum_{w=0}^{W-1} y_{w,c} - 1\right)^2
$$

Expanding this gives:

$$
P \sum_{w=0}^{W-1} y_{w,c}^2 - 2P \sum_{w=0}^{W-1} y_{w,c} + P
$$

Which is implemented in the model using:
- **Linear terms**: $ -2P \cdot y_{w,c} $  
- **Quadratic terms**: $ +2P \cdot y_{w1,c} \cdot y_{w2,c} $ for all warehouse pairs $ w1 \neq w2 $  
- **Offset term**: $ +P $, added once per customer  

This ensures that if a customer is assigned **more than one warehouse**, the penalty increases significantly, discouraging invalid assignments.

---

### **4. Bias Term: Preferential Warehouse Assignment**
To **bias** the assignment process towards a particular warehouse (e.g., `Warehouse #0`), a **negative linear term** is added for all $ y_{0,c} $ variables:

$$
B \sum_{c=0}^{C-1} y_{0,c}
$$

Where $ B $ is a negative bias value (e.g., $ B = -5 $). This **reduces the cost of assigning customers to Warehouse #0**, making it more likely to be selected when the solver optimizes the assignments.

---

### **5. Solving the QUBO Model**
The QUBO model is solved using **Simulated Annealing**, a probabilistic optimization algorithm designed for combinatorial problems. It iteratively refines candidate solutions to find a configuration that minimizes the total cost while satisfying constraints.

The solver produces a **binary assignment matrix**, where each $ y_{w,c} = 1 $ indicates an active warehouse-to-customer assignment.

---

### **6. Visualization: Scaled Warehouse-Customer Network**
The final assignment solution is visualized using **NetworkX**. The visualization includes:  
- **Red squares** (`W0, W1, ...`) representing **warehouses**  
- **Green circles** (`C0, C1, ...`) representing **customers**  
- **Edges (links)** connecting warehouses to customers  
  - **Blue edges**: Selected warehouse-customer assignments in the final optimized solution  
  - **Gray edges**: Possible assignments that were **not selected**  

To ensure that **distances are reflected accurately**, the visualization uses a **force-directed layout (`kamada_kawai_layout`)**, which arranges nodes so that:
- **Shorter distances** → **Closer nodes**  
- **Longer distances** → **Further apart nodes**  

This provides an **intuitive spatial representation** of warehouse-customer relationships based on the **actual numerical distances**.

---

### **7. Summary of Features**
✔ **Minimizes total transport costs** using a distance-based objective function  
✔ **Ensures each customer gets exactly one warehouse** using a penalty-based constraint  
✔ **Allows warehouse preference** through a **bias term**  
✔ **Solves using Simulated Annealing** to find the best assignments  
✔ **Visualizes the optimal solution** with a distance-scaled network graph  

---

### **8. Possible Extensions**
🔹 **Add Warehouse Capacity Constraints**: Prevent exceeding storage limits at each warehouse  
🔹 **Multi-Objective Optimization**: Introduce additional cost factors (e.g., priority customers)  
🔹 **Quantum Optimization**: Run the model on a quantum annealer (e.g., D-Wave) for faster performance  

This QUBO model provides a **strong foundation** for solving warehouse allocation problems and can be extended for larger, real-world supply chain applications. 


## Contents of `data` Folder

Contains data files which will be fed into the data set

## Contents of `components` Folder

Once the model is further developed, it will have several important components which can be accessed through this folder

## Contents of `utils` Folder

Contains extraneous files which are needed for various operations in this project