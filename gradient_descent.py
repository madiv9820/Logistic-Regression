import numpy    # For n-d arrays and mathematical operations

'''
    A logistic regression code from scratch, using gradient
    descent technique. Code has been optimized using various
    mathematical numpy functions. 
'''
class LogisticRegression:
    # Constructor
    def __init__(self,
                 alpha: numpy.int32= 1, 
                 iterations: numpy.int32 = 1000, 
                 tolerance: numpy.float64 = 0.0001) -> None:
        '''
            alpha: Learning rate
            iterations: total no of counts of training 
                        through gradient descent
            tolerance: maximum difference of the cost between the iteration,
                        if less then will stop training
        '''
        self.__alpha = alpha
        self.__iterations = iterations
        self.__tolerance = tolerance

    # String Representation of the class
    def __repr__(self) -> str:
        return (f'gradient_descent.LogisticRegression' + 
                f'< alpha = {self.__alpha}, ' + 
                f'iterations = f{self.__iterations}, ' +  
                f'tolerance = f{self.__tolerance} >')
    
    '''
        Sigmoid Function
        Sig(z) = 1 / (1 + e^(-z))
        z = x.m
    '''
    def __sigmoid_function(self, 
                           x: numpy.ndarray, 
                           m: numpy.ndarray) -> float:
        z = numpy.dot(x, m)
        return 1/(1 + numpy.exp(-z))

    '''
        Cost Function
        cost = Mean from i = 1 to n ((-y[i] * log(sigmoid(z)) 
                                        - (1-y[i]) * log(1 - sigmoid(z)))
    '''
    def __cost(self, 
                x: numpy.ndarray, 
                y: numpy.ndarray, 
                m: numpy.ndarray) -> float:
        sigmoid_Values = self.__sigmoid_function(x, m)
        sigmoid_Values = numpy.clip(sigmoid_Values, 1e-10, 1 - 1e-10)
        
        cost = -numpy.mean(y * numpy.log(sigmoid_Values) + 
                            (1-y) * numpy.log(1 - sigmoid_Values))
        
        return cost
    
    '''
        Step Gradient Function, returns slop for each feature
        d(Cost)/d(feature jth) = Mean from i = 1 to n, (y[i] - sigmoid(z)) * feature[i, j]
    '''
    def __step_gradient_descent(self, 
                                x: numpy.ndarray, 
                                y: numpy.ndarray, 
                                m: numpy.ndarray) -> numpy.ndarray:
        
        predictions = self.__sigmoid_function(x, m)
        errors = predictions - y
        gradients = numpy.dot(x.T, errors) / x.shape[0]
        
        return gradients
    

    # Gradient Function, to train the algorithm
    def __gradient_descent(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        # Creating constants with some random values
        self.__constants = numpy.random.rand(x.shape[1])
        
        for count in range(self.__iterations):
            # Getting slope after each gradient descent process
            slopes = self.__step_gradient_descent(x, y, self.__constants)
            # Creating new constants
            new_constants = self.__constants - self.__alpha * slopes

            # Calculating costs using new constants and current constants
            previous_Cost = self.__cost(x, y, self.__constants)
            current_Cost = self.__cost(x, y, new_constants)

            # print('Iteration:', count+1, end = ' ')
            # print('Previous Cost:', previous_Cost, end = ' ')
            # print('Current Cost:', current_Cost, end = ' ')
            # print('Alpha:', self.__alpha)
            
            # If current cost > previous cost, then learning rate
            # is high need to reduce it, else updating constants
            if current_Cost > previous_Cost: self.__alpha /= 2
            else: self.__constants = new_constants

            # If difference between costs is less than tolerance,
            # we can stop training the algorithm
            if abs(previous_Cost - current_Cost) < self.__tolerance: break
            
            # After 100 steps, we are increasing alpha, since it is
            # taking a lot of time to reach minima
            if count != 0 and count % 100 == 0: self.__alpha *= 2
    
    # Fit function to train the algorithm
    def fit(self, x: numpy.ndarray, y: numpy.ndarray) -> None:
        x = numpy.append(x, numpy.ones((x.shape[0], 1)), axis = 1)
        self.__gradient_descent(x, y)

        self.coef_ = self.__constants[:-1]
        self.intercept_ = self.__constants[-1]
    
    # Predict function, which returns predicted class
    def predict(self, x: numpy.ndarray) -> numpy.ndarray:
        x = numpy.append(x, numpy.ones((x.shape[0], 1)), axis = 1)
        y_pred = self.__sigmoid_function(x, self.__constants)
        return numpy.where(y_pred <= 0.5, 0, 1)
    
    # Score function, which calculates mean accuracy
    def score(self, x: numpy.ndarray, y: numpy.ndarray) -> float:
        y_pred = self.predict(x)
        equal = numpy.sum(y_pred == y)
        return equal / y.shape[0]
    
    # Returns probability of each class for each row
    def predict_proba(self, x: numpy.ndarray) -> numpy.ndarray:
        x = numpy.append(x, numpy.ones((x.shape[0], 1)), axis = 1)
       
        prob_class_1 = self.__sigmoid_function(x, self.__constants)
        prob_class_0 = 1 - prob_class_1

        return numpy.column_stack((prob_class_0, prob_class_1))