def em(events, T, theta_guess = [0.5,0.5,0.5], steps=1000, learning_rate=0.01):
    mu,alpha,beta = theta_guess
    n = len(events)
    times = np.array([e[0] for e in events])
    
    momentum = 0
    loglikelihood_opt = None
    theta_opt = None
    for _ in range(steps):
        print("E")
        #E step
        branching_matrix = np.array([[0 for _ in range(len(events)+1)]]+[[mu]+[alpha*np.exp(beta*(e2[0]-e[0])) if e2[0]<e[0] else 0 for e2 in events] for e in events])
        print("Normalise")
        c = branching_matrix.sum(axis=1)
        c[0] = 1
        branching_matrix /= c.reshape(-1,1)
        
        plt.imshow(branching_matrix[:30,:30])
        plt.show()
        
        logbeta = np.log(beta)
        
        print("M")
        #M step
        @jax.jit
        def q(logbeta):
            beta = jnp.exp(logbeta)
            alpha = beta * branching_matrix[1:,1:].sum() / (1-jnp.exp(beta*(times-T))).sum()
            mu = (1/T) * branching_matrix[1:,0].sum()
            sum_residuals = mu*T + (alpha/beta)*(1-jnp.exp(beta*(times-T))).sum()
            #assert np.allclose([branching_matrix[i,0]+sum([branching_matrix[i,j] for j in range(1,i)]) for i in range(1,n+1)],1)
            sum_expected_event_ll = (branching_matrix[1:n+1,0]*np.log(mu)).sum() + (branching_matrix[1:,1:]*(jnp.log(alpha)+beta*(times.reshape(-1,1)-times.reshape(1,-1)))).sum()
            return sum_expected_event_ll - sum_residuals
        
        print("q")
        loglikelihood = q(logbeta)
#         gradient_step = np.array(jax.grad(q)(logbeta))
#         step_size = learning_rate * np.linalg.norm(gradient_step)**-2
#         logbeta += step_size * gradient_step

        print("Newton Step")
        newton_step = -jax.grad(q)(logbeta) / jax.hessian(q)(logbeta)
        logbeta += newton_step
        
        print("Beta,mu,alpha")
        beta = np.exp(logbeta)
        mu = (1/T) * branching_matrix[1:,0].sum()
        alpha = beta * branching_matrix[1:,1:].sum() / (1-jnp.exp(beta*(times-T))).sum()
        
        print("Update loglikelihood_opt")
        if loglikelihood_opt is None or loglikelihood > loglikelihood_opt:
            theta_opt = mu,alpha,beta
            loglikelihood_opt = loglikelihood
        
        #print(f'{loglikelihood=}\n{mu,alpha,beta=}\n{gradient_step*step_size=}\n')
        print(f'{loglikelihood=}\n{mu,alpha,beta=}\n{newton_step=}\n')
        
        """
        newton_step = -np.linalg.solve(jax.hessian(q)([logalpha,beta]), jax.grad(q)([logalpha,beta]))
        gradient_step = np.array(jax.grad(q)([logalpha,beta]))
        print(f'{logalpha,beta=}, {newton_step=}, {gradient_step=}')
        if all((newton_step*gradient_step)>0):
            print('newton')
            logalpha,beta = np.array([logalpha,beta]) + newton_step
        else:
            print('gradient')
            logalpha,beta = np.array([logalpha,beta]) + step_size * gradient_step
        """
        
        
    return theta_opt
