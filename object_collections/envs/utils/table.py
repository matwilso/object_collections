import numpy as np
import quaternion

TABLES = {}
TABLES['big'] = BIG = {}
TABLES['small'] = SMALL = {}
TABLES['robot'] = ROBOT = {}
TABLES['folding'] = FOLDING = {}
# height from tabletop to floor
BIG['height'] = 0.89
SMALL['height'] = 0.5975
ROBOT['height'] = 0.5975
FOLDING['height'] = 0.74
# dims of tabletop
BIG['wood'] = np.array([0.76, 1.22, 0.045])
SMALL['wood'] = np.array([0.615, 0.92, 0.035])
ROBOT['wood'] = np.array([0.61, 0.9175, 0.035]) 
FOLDING['wood'] = np.array([0.6, 1.21, 0.04]) 
# delta from robot floor / base, center to center
BIG['pos'] = np.array([-0.88, 0.0, BIG['height'] - BIG['wood'][2]]) 
SMALL['pos'] = np.array([-0.76, 0.0, SMALL['height'] - SMALL['wood'][2]])
ROBOT['pos'] = np.array([0.0, 0.0, ROBOT['height'] - ROBOT['wood'][2]])
FOLDING['pos'] = np.array([-0.57, 0.0, FOLDING['height'] - FOLDING['wood'][2]])
# width of table legs
BIG['leg_width'] = 0.05
SMALL['leg_width'] = 0.035
ROBOT['leg_width'] = 0.035
FOLDING['leg_width'] = 0.035  # TODO:

ELLIPSOID = 4
CYLINDER = 5
BOX = 6

# leg geom type
BIG['leg_gtype'] = BOX
SMALL['leg_gtype'] = BOX
ROBOT['leg_gtype'] = BOX
FOLDING['leg_gtype'] = CYLINDER

# whether table has xbar going along the longer axis (big table doesn't have it)
BIG['lengthwise_xbar'] = False
SMALL['lengthwise_xbar'] = True
ROBOT['lengthwise_xbar'] = True
FOLDING['lengthwise_xbar'] = False
# from edge of table to xbar/leg
BIG['xbar_offset'] = 0.07
SMALL['xbar_offset'] = 0.01
ROBOT['xbar_offset'] = 0.01
FOLDING['xbar_offset'] = 0.075
# from bottom of table wood to top of xbar
BIG['wood_to_xbar'] = 0.58
SMALL['wood_to_xbar'] = 0.28
ROBOT['wood_to_xbar'] = 0.28
FOLDING['wood_to_xbar'] = 0.60

def set_table(model, geom_name, spec_name, FLAGS):
    """code to programmatically build a table from boxes, based on parameters"""
    name2bid = model.body_name2id
    name2gid = model.geom_name2id

    bid = name2bid(geom_name)
    gid = name2gid(geom_name)

    table_dims = TABLES[spec_name]

    model.body_pos[bid] = table_dims['pos']
    model.geom_pos[gid] = np.array([0,0,table_dims['wood'][2] / 2])
    model.geom_size[gid] = table_dims['wood'] / 2

    if spec_name == 'robot':
        # special case, fix the base_link
        base_link_bid = name2bid('base_link')
        model.body_pos[base_link_bid] = np.array([0, 0, table_dims['wood'][2]/2 + 0.0275])
        model.body_quat[bid] = quaternion.from_euler_angles(np.pi/2,0,0).components
    
    #if FLAGS['baxter']:
    #    model.body_pos[bid] = model.geom_pos[name2gid('ground')] + table_dims['pos']

    # LEGS
    leg_bid = lambda i: name2bid(geom_name+'_leg'+str(i))
    leg_gid = lambda i: name2gid(geom_name+'_leg'+str(i))

    xpos = lambda dir: dir*table_dims['wood'][0]/2 - dir*table_dims['xbar_offset'] - dir*table_dims['leg_width']/2
    ypos = lambda dir: dir*table_dims['wood'][1]/2 - dir*table_dims['xbar_offset'] - dir*table_dims['leg_width']/2
    leg_xsection = table_dims['leg_width']/2

    i = 0
    for xdir in [-1,1]:
        for ydir in [-1,1]:
            model.body_pos[leg_bid(i)] = np.array([xpos(xdir), ypos(ydir), -table_dims['pos'][2]/2])
            model.geom_size[leg_gid(i)] = np.array([leg_xsection, leg_xsection, table_dims['pos'][2]/2])
            if table_dims['leg_gtype'] == CYLINDER:
                model.geom_size[leg_gid(i)][:2] = model.geom_size[leg_gid(i)][1:3] 
                model.geom_type[leg_gid(i)] = table_dims['leg_gtype']
            i += 1
    # XBARS
    xbar_bid = lambda i: name2bid(geom_name+'_bar'+str(i))
    xbar_gid = lambda i: name2gid(geom_name+'_bar'+str(i))
    heights = [-table_dims['leg_width']/2, -table_dims['leg_width']/2 - table_dims['wood_to_xbar']] # top xbar, lower xbar
    
    i = 0 
    for ydir in [-1,1]:
        for h in heights:
            model.body_pos[xbar_bid(i)] = np.array([0.0, ypos(ydir), h])
            length = (table_dims['wood'][0] - 2*table_dims['xbar_offset']) / 2
            model.geom_size[xbar_gid(i)] = np.array([length, table_dims['leg_width']/2, table_dims['leg_width']/2])
            if table_dims['leg_gtype'] == CYLINDER:
                #model.geom_size[xbar_gid(i)][:2] = model.geom_size[xbar_gid(i)][1:3] 
                model.geom_type[xbar_gid(i)] = ELLIPSOID#table_dims['leg_gtype']
            i += 1

    if table_dims['lengthwise_xbar']:
        # LENGTHWISE XBARS
        lbar_bid = lambda i: name2bid(geom_name+'_lbar'+str(i))
        lbar_gid = lambda i: name2gid(geom_name+'_lbar'+str(i))

        heights = [-table_dims['leg_width']/2, -table_dims['leg_width']/2 - table_dims['wood_to_xbar']] # top xbar, lower xbar
        i = 0 
        for xdir in [-1,1]:
            for h in heights:
                model.body_pos[lbar_bid(i)] = np.array([xpos(xdir), 0.0, h])
                length = (table_dims['wood'][1] - 2*table_dims['xbar_offset']) / 2
                model.geom_size[lbar_gid(i)] = np.array([table_dims['leg_width']/2, length, table_dims['leg_width']/2])
                model.geom_type[xbar_gid(i)] = table_dims['leg_gtype']
                i += 1