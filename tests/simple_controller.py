def simple_control(obs):
        '''
        0 means do nothing, 1 means flap
        '''
        c = -15 # from paper
        for pipe in obs['pipes']:
            if pipe['x'] - obs['x'] > 0:
                d = (pipe['y'] - 50) - obs['y']
                break
        print(f'd = {d}')
        if d < c:
            action = 1
        else:
            action = 0  

        return action