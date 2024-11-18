import numpy as np

def div(center, neighbours, field):
    
    for coordinate in neighbours.keys():
        for direction in neighbours[coordinate].keys():
            if neighbours[coordinate][direction] == None or np.isnan(neighbours[coordinate][direction][field+'r']):
                neighbours[coordinate][direction] = center
    
    th0 = center["th_c"]; r0 = center["r_c"];
    
    divergence = 1/r0*(2*center[field + "r"] \
               + (neighbours["th"][+1][field + "th"] - neighbours["th"][-1][field + "th"])/(neighbours['th'][+1]['th_c'] - neighbours['th'][-1]['th_c'])) + 1/(r0*np.sin(th0))*(np.cos(th0)*center[field + 'th'] \
               + (neighbours['phi'][+1][field + 'phi'] - neighbours['phi'][-1][field + 'phi'])/(neighbours['phi'][+1]['phi_c'] - neighbours['phi'][-1]['phi_c'])) \
               + (neighbours['r'][+1][field + 'r'] - neighbours['r'][-1][field + 'r'])/(neighbours['r'][+1]['r_c'] - neighbours['r'][-1]['r_c'])
    
    return divergence

def curl(center, neighbours, field):
        
    for coordinate in neighbours.keys():
        for direction in neighbours[coordinate].keys():
            if neighbours[coordinate][direction] == None or np.isnan(neighbours[coordinate][direction][field+'r']):
                neighbours[coordinate][direction] = center
               #return [np.nan, np.nan, np.nan]
            #else:
            
#     th0 = center.th_c; r0 = center.r_c; 

#     curl_r = np.cos(th0)/(r0*np.sin(th0))*getattr(center, field + 'phi') \
#            + 1/r0 * ((getattr(neighbours['th'][+1], field+'phi') - getattr(neighbours['th'][-1], field+'phi'))/(getattr(neighbours['th'][+1], 'th_c') - getattr(neighbours['th'][-1], 'th_c'))) \
#            - 1/(r0*np.sin(th0)) * ((getattr(neighbours['phi'][+1], field+'th') - getattr(neighbours['phi'][-1], field+'th'))/(getattr(neighbours['phi'][+1], 'phi_c') - getattr(neighbours['phi'][-1], 'phi_c')))

#     curl_th = 1/(r0*np.sin(th0))*((getattr(neighbours['phi'][+1],field+'r')-getattr(neighbours['phi'][-1],field+'r'))/(getattr(neighbours['phi'][+1],'phi_c')-getattr(neighbours['phi'][-1], 'phi_c'))) \
#             - getattr(center, field+'phi')/r0 \
#             - (getattr(neighbours['r'][+1],field+'phi') - getattr(neighbours['r'][-1],field+'phi'))/(getattr(neighbours['r'][+1],'r_c') - getattr(neighbours['r'][-1],'r_c'))

#     curl_phi = getattr(center,field+'th')/r0 \
#              + (getattr(neighbours['r'][+1],field+'th')-getattr(neighbours['r'][-1],field+'th'))/(getattr(neighbours['r'][+1],'r_c')-getattr(neighbours['r'][-1],'r_c')) \
#              - 1/r0 * ((getattr(neighbours['th'][+1],field+'r') - getattr(neighbours['th'][-1], field+'r'))/(getattr(neighbours['th'][+1], 'th_c') - getattr(neighbours['th'][-1], 'th_c')))
    
    
    th0 = center["th_c"]; r0 = center["r_c"]; 

    curl_r = np.cos(th0)/(r0*np.sin(th0))*center[field + 'phi'] \
           + 1/r0 * (neighbours['th'][+1][field+'phi'] - neighbours['th'][-1][field+'phi'])/(neighbours['th'][+1]['th_c'] - neighbours['th'][-1]['th_c']) \
           - 1/(r0*np.sin(th0)) * (neighbours['phi'][+1][field+'th'] - neighbours['phi'][-1][field+'th'])/(neighbours['phi'][+1]['phi_c'] - neighbours['phi'][-1]['phi_c'])

    curl_th = 1/(r0*np.sin(th0))*(neighbours['phi'][+1][field+'r'] - neighbours['phi'][-1][field+'r'])/(neighbours['phi'][+1]['phi_c'] - neighbours['phi'][-1]['phi_c']) \
            - center[field+'phi']/r0 \
            - (neighbours['r'][+1][field+'phi'] - neighbours['r'][-1][field+'phi'])/(neighbours['r'][+1]['r_c'] - neighbours['r'][-1]['r_c'])

    curl_phi = center[field+'th']/r0 \
             + (neighbours['r'][+1][field+'th'] - neighbours['r'][-1][field+'th'])/(neighbours['r'][+1]['r_c'] - neighbours['r'][-1]['r_c']) \
             - 1/r0 * (neighbours['th'][+1][field+'r'] - neighbours['th'][-1][field+'r'])/(neighbours['th'][+1]['th_c'] - neighbours['th'][-1]['th_c'])

    return [curl_r, curl_th, curl_phi]

def sigma_gradient(center, neighbours, field, comp, resp):
    
    for coordinate in neighbours.keys():
        for direction in neighbours[coordinate].keys():
            if neighbours[coordinate][direction] == None or np.isnan(neighbours[coordinate][direction][field+'r']):
                neighbours[coordinate][direction] = center

    sigma_gr = np.sqrt(neighbours[resp][-1]['sigma_'+field+comp]**2 + 2*center['sigma_'+field+comp]**2 + neighbours[resp][+1]['sigma_'+field+comp]**2)/(np.sqrt(2)*center['d'+resp])

    return sigma_gr