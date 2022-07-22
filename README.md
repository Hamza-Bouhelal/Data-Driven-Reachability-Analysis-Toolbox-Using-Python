# Data-Driven-Reachability-Analysis-Toolbox-Using-Python
![alt text](https://github.com/Hamza-Bouhelal/Data-Driven-Reachability-Analysis-Toolbow-Using-Python/blob/main/NonLinear%20sys%20Plot/%20Plot%20n1.png?raw=true)
<br/>
## Setup: 
### Git
- Initilize a repository in an empty directory:  ```git init``` <br/>
- run: ```git clone https://github.com/Hamza-Bouhelal/Data-Driven-Reachability-Analysis-Toolbox-Using-Python``` <br/>
- use ```git status``` to check whether everything is up to date <br/>
### Python Dependencies
- Python & pip
- Zonotope Set representation Implementation
- ```cd Data-Driven-Reachability-Analysis-Toolbox-Using-Python```
- ```pip install virtualenv```
- ```virtualenv env```
- ```.\venv\Scripts\activate```
- ```pip install -r requirements.txt```

## Example: 
```
    from Linear_sys import linear_sys
    import numpy as np
    from Zonotope import Zonotope
    steps = 100
    initpoints = 1
    dim_x = 3
    X0 = Zonotope(np.array(np.ones((dim_x, 1))), 0.15 *np.diag(np.ones((dim_x, 1)).T[0]))
    U = Zonotope(10, 0.25)
    W = Zonotope(np.array(np.zeros((dim_x, 1))), 0.005 * np.ones((dim_x, 1)))
    A = np.array([[1, 0, 0], [1, 1, 0],[0, 1, 1]])
    B_ss = np.array([1, -1, 0])
    C = np.array([1, 0, 0])
    D = 0
    L_sys = linear_sys(A, B_ss, C, D, X0, U, W, dim_x, initpoints, steps, 0.05)
    model, X_data = L_sys.Run_reachability(6, plot=False, save="LinearSys Plot")
```


