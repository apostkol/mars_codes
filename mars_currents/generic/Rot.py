import numpy as np

def Rot(th, axis = 'x'):
    if axis== 'x':
        rot_m = [[1,          0,           0],
                 [0, np.cos(th), -np.sin(th)],
                 [0, np.sin(th),  np.cos(th)]]
    elif axis == 'y':
        rot_m = [[np.cos(th), 0,  np.sin(th)],
                 [0,          1,           0],
                 [-np.sin(th), 0, np.cos(th)]]
    elif axis == 'z':
        rot_m = [[np.cos(th), -np.sin(th), 0],
                 [np.sin(th),  np.cos(th), 0],
                 [0,           0,          1]]
    return rot_m
    
    # return np.matrix([[-np.cos(th)*np.cos(phi), np.cos(th)*np.sin(phi), np.sin(th)],
    #                   [-np.sin(th)*np.cos(phi), -np.sin(th)*np.sin(phi), np.cos(th)],
    #                   [-np.sin(phi), np.cos(phi), 0]])
    
#def sph_to_cart_transf():
    
    
# def sph_to_cart_vf(A, th, phi):
    
#     #th = np.deg2rad(th); phi = np.deg2rad(phi)
#     if isinstance(th, list) and isinstance(phi, list):
        
#         points_rot = [cart_to_sph_transf(theta, phii).T for theta, phii in zip(th, phi)]
#     #print(points_rot[0])
#         rotated_to_cart = [np.dot(rot_m, point) for rot_m, point in zip(points_rot, A)]
#     else:
        
#         rot_m = cart_to_sph_transf(th, phi).T
#         rotated_to_cart = np.dot(rot_m, A)
        
#     return rotated_to_cart

# def cart_to_sph_vf(A, th, phi):
    
#     #th = np.deg2rad(th); phi = np.deg2rad(phi)
    
#     points_rot = [cart_to_sph_transf(theta, phi) for theta, phi in zip(th, phi)]
#     rotated_to_sph = [np.dot(rot_m, point) for rot_m, point in zip(points_rot, A)]
    
#     return rotated_to_sph

