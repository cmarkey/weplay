import joblib
import numpy as np
# import pandas as pd
from scipy.stats import norm
from numpy.typing import ArrayLike
from sklearn.ensemble import RandomForestRegressor
# import pickle #Robyn - to load in RF model
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance_matrix

MAX_TIME = 2
EPS = 1e-7
GG = 32.174

METRICS = ['prob', 'rink_ctrl', 'best_case', 'expected']
TIME_PENALTY = 0.1
MAX_VEL = 35.5   # maximum skater velocity in ft/sec
# acceleration coefficient (not directly acceleration, but more like a speed decay)
ALPHA = 1.3
TR = 0.189  # reaction time (based on the article Phil sent)
MM = 0.1  # Coefficient of friction between puck and ice, I'll find the source for this
# Puck air drag coefficient (actuall it's the coefficient divided by the mass so beta = k/m if k is the drag coefficient)
BETA_PUCK = 0.1322
BETA_CTRL = 2.5  # pitch control coefficient used as beta in ice_ctrl_xyt and teamwise_ice_ctrl_xyt, taken from the Spearman paper
X_DECAY = 2000  # value used as decay_x
Y_DECAY = 500  # value used as decay_y
GOALIE_DIST = 8  # maximum reasonable distance for goalie to go away from goal
GLX = 11  # Goalie X coord
GLY = 42.5  # Goalie Y coord
STICK = 5  # Stick length
TARGET_RADIUS = 28
LOADED_RF = joblib.load(
    'https://github.com/cmarkey/weplay/raw/main/Passing%20Analysis/code/loaded_rf.joblib')


def test():
    print(1+2)


def inside_boards(x: np.ndarray, y: np.ndarray, t: np.ndarray,
                  target_radius: float = TARGET_RADIUS):
    radius = (x < 28) * ((y > 57)*((x-28)**2 + (y-57)**2) **
                         0.5 + (y < 28)*((x-28)**2 + (28-y)**2)**0.5)
    ix = (radius <= target_radius) * (0 < x) * (x < 100) * (0 < y) * (y < 85)
    return x[ix], y[ix], t[ix]


class pass_tracks():
    def __init__(self,
                 # x locations of players (array or list of floats)
                 x: ArrayLike,
                 # y locations of players (array or list of floats)
                 y: ArrayLike,
                 # x velocity of players (array or list of floats)
                 vx: ArrayLike,
                 # y velocity of players (array or list of floats)
                 vy: ArrayLike,
                 goalie: int,  # column number for goalie
                 puck: int,  # column number for which player has the puck
                 # array or list of integers +1 for offence, -1 for defence (or true and false)
                 off: ArrayLike,
                 vp: float = 55,
                 phi_res: float = 0.01,
                 t_res: float = 0.01,
                 # metric: str = 'expected'
                 ):
        assert len(set([len(x), len(y), len(vx), len(vy), len(off)])) <= 2
        # if not metric in METRICS:
        # raise ValueError('Metric choice is not in recognized metric list, please choose another metric')
        # self.metric = metric
        self.puck = int(puck)
        self.xp = x[self.puck]
        self.yp = y[self.puck]
        self.phi_res = float(phi_res)
        self.off = np.where(np.array(list(off)) == 1, 1, -1)
        self.t_res = float(t_res)
        self.vp = float(vp)
        self.x = np.array(list(x))
        self.y = np.array(list(y))
        self.vx = np.array(list(vx))
        self.vy = np.array(list(vy))
        self.goalie = int(goalie)
        # self.tracks = pd.DataFrame({'x':x,'y':y,'vx':vx,'vy':vy,'goalie':goalie,'off':off})
        self.player_motion()
        # self.grid = np.concatenate([self.one_pass(self, phi)() for phi in np.arange(-np.pi,np.pi+EPS, phi_res)], axis = 0)
        full_grid = [self.one_pass(self, phi)()
                     for phi in np.arange(-np.pi, np.pi+EPS, phi_res)]
        self.grid = np.concatenate([fg['grid'] for fg in full_grid], axis=0)
        self.triangles = np.stack([fg['triangle'] for fg in full_grid], axis=0)
        self.domains = [[min(x), max(x)] for x in self.triangles.T]

    def player_motion(self, alpha: float = ALPHA, t_r: float = TR, vmax: float = MAX_VEL):
        t = np.arange(self.t_res, MAX_TIME, self.t_res).reshape(-1, 1)
        # x = self.x.reshape(-1,1)
        # vx = self.vx.reshape(-1,1)
        # y = self.y.reshape(-1,1)
        # vy = self.vy.reshape(-1,1)

        self.c_x = np.where(t < t_r, self.x + self.vx * t, self.x +
                            t_r * self.vx + self.vx * (1-np.exp(-alpha * (t-t_r))/alpha))
        self.c_y = np.where(t < t_r, self.y + self.vy * t, self.y +
                            t_r * self.vy + self.vy * (1-np.exp(-alpha * (t-t_r))/alpha))
        self.r = np.where(t < t_r, 0, vmax * (t - t_r -
                          (1-np.exp(-alpha * (t-t_r)))/alpha))

    class one_pass():
        def __init__(self, outer_self: 'tracks', phi: float):
            self.t_res = outer_self.t_res
            self.phi = phi
            self.x0 = outer_self.xp
            self.y0 = outer_self.yp
            self.vp = outer_self.vp
            self.x, self.y, self.t = self.make_grid()
            self.outside_creese = (self.x-GLX)**2 + \
                (self.y-GLY)**2 > GOALIE_DIST**2
            self.get_metric(outer_self)  # , outer_self.metric)

        def make_grid(self):
            t = np.arange(self.t_res, MAX_TIME, self.t_res)
            x, y = self.puck_motion_model(t)
            return inside_boards(x, y, t)

        def puck_motion_model(self, t: np.ndarray,
                              mu: float = MM,
                              beta: float = BETA_PUCK,
                              g: float = GG):
            vx = self.vp*np.sin(self.phi)
            vy = -self.vp*np.cos(self.phi)

            x = self.x0 + (vx + mu*g * vx/self.vp/beta) * (1 -
                                                           np.exp(-beta * t))/beta - (mu*g * t * vx/self.vp)/beta
            y = self.y0 + (vy + mu*g * vy/self.vp/beta) * (1 -
                                                           np.exp(-beta * t))/beta - (mu*g * t * vy/self.vp)/beta

            return x, y

        def score_prob(self, decay_x=X_DECAY, decay_y=Y_DECAY):
            # Scoring Probability function
            x = self.x
            y = self.y
            self.score = (np.abs((x-GLX)/((GLY-y)**2+(GLX-x)**2)**0.5)+1)/np.where(
                x < GLX, 8, 4)*np.exp(-((GLX-x)**2/decay_x + (GLY-y)**2/decay_y))

        def dist_to_xyt(self, outer_self: 'tracks'):
            # If time is smaller than reaction time, skater keeps going at initial speed
            ln = len(self.t)
            tx = self.x.reshape(-1, 1)
            ty = self.y.reshape(-1, 1)

            remaining_dist = (
                (tx-outer_self.c_x[:ln, :])**2 + (ty-outer_self.c_y[:ln, :])**2)**0.5-outer_self.r[:ln, :]
            return(np.maximum(remaining_dist, EPS))

        def get_metric(self, outer_self: 'tracks'):  # , metric: str = 'prob'):
            # dists = np.array([self.dist_to_xyt(x0,y0,vx,vy) for x0,y0,vx,vy in zip(outer_self.x, outer_self.y, outer_self.vx, outer_self.vy)]).T
            dists = self.dist_to_xyt(outer_self)
            dists[self.outside_creese, outer_self.goalie] = np.maximum(dists[self.outside_creese, outer_self.goalie], ((
                self.x[self.outside_creese]-GLX)**2 + (self.y[self.outside_creese]-GLY)**2)**0.5 - GOALIE_DIST)
            ctrl = (dists/MAX_VEL)**(-BETA_CTRL) * \
                outer_self.off.reshape(1, -1)
            self.all_ctrl = ctrl.sum(1)/np.abs(ctrl).sum(1)
            # if metric == 'rink_ctrl':
            # self.metric = self.all_ctrl
            # return 0

            dists = np.delete(dists, outer_self.puck, axis=1)
            off_mat = np.delete(outer_self.off, outer_self.puck)
            base_probs = self.t_res * (norm.cdf(dists/STICK+1)-norm.cdf(dists/STICK-1))/TIME_PENALTY * (
                1 - np.exp(-self.t.reshape(-1, 1)/(TR + TIME_PENALTY * off_mat.reshape(1, -1))))
            ranks = (-base_probs).argsort()
            # print(ranks.shape, base_probs.shape)
            ranked_probs = np.take_along_axis(base_probs, ranks, 1)
            off_mat = off_mat[ranks]
            # print(off_mat)
            # print(np.concatenate((np.ones((1,dists.shape[1])),1-ranked_probs[:-1,:]),0).cumprod(1).shape)
            adj_probs = np.concatenate(
                (np.ones((1, dists.shape[1])), 1-ranked_probs[:-1, :]), 0).cumprod(1)*ranked_probs
            # print(adj_probs.shape)
            adj_pass_off = (adj_probs*(off_mat == 1)).sum(1)
            # pass_def = adj_probs[~off_mat] # Not sure we actually need this line
            missed = 1 - adj_probs.sum(1)
            missed = np.append(1, missed[:-1]).cumprod()
            pass_off = adj_pass_off * missed
            # if metric == 'prob':
            #     self.metric = pass_off.sum()  * np.ones(self.x.shape)
            # if metric == 'best_case':
            #     self.score_prob()
            #     adj_pass_value = self.score*all_ctrl*adj_pass_off
            #     self.metric = adj_pass_value.max() * np.ones(self.x.shape)

            # elif metric == 'expected':
            #     self.score_prob()
            #     loc_pass_value = self.score*all_ctrl*pass_off
            #     self.metric = loc_pass_value.sum() * np.ones(self.x.shape)
            dr = ((self.x[-1]-outer_self.xp)**2 + (self.y[-1] -
                  outer_self.yp)**2)**0.5 * outer_self.phi_res/2
            self.prob = pass_off.sum()
            self.score_prob()
            adj_pass_value = self.score*self.all_ctrl*adj_pass_off
            self.best_case = adj_pass_value.max()
            loc_pass_value = self.score*self.all_ctrl*pass_off
            self.expected = loc_pass_value.sum()
            self.triangle_metrics = [self.x[-1] + dr * np.cos(self.phi), self.y[-1] + dr * np.sin(self.phi), self.x[-1] - dr * np.cos(
                self.phi), self.y[-1] - dr * np.sin(self.phi), self.prob, self.best_case, self.expected, self.phi]

        def __call__(self):
            # Robyn - added self.score*self.all_ctrl for location value of passer
            return {'grid': np.stack((self.x, self.y, self.t, self.all_ctrl, self.score*self.all_ctrl), 1), 'triangle': self.triangle_metrics}

# Robyn - added metrics which uses tracks to calculate various metrics used in modelling


class metrics(pass_tracks):
    # don't know how to initialize properly in python yet so this is my quick fix
    # ideally all these will move to within tracks once we know which we want to keep
    def __init__(self,
                 # x locations of players (array or list of floats)
                 x: ArrayLike,
                 # y locations of players (array or list of floats)
                 y: ArrayLike,
                 # x velocity of players (array or list of floats)
                 vx: ArrayLike,
                 # y velocity of players (array or list of floats)
                 vy: ArrayLike,
                 goalie: int,  # column number for goalie
                 puck: int,  # column number for which player has the puck
                 # array or list of integers +1 for offence, -1 for defence (or true and false)
                 off: ArrayLike,
                 vp: float = 55,
                 phi_res: float = 0.01,
                 t_res: float = 0.01,
                 #  rf=LOADED_RF
                 # metric: str = 'expected'
                 ):
        super().__init__(x, y, vx, vy, goalie, puck, off, vp, t_res)

        self.xgrid = self.grid[:, 0]
        self.ygrid = self.grid[:, 1]
        self.rcgrid = self.grid[:, 3]
        # self.locvalgrid = self.grid[:,4]
        # self.successtriangle = self.triangles[:,4]
        # self.besttriangle = self.triangles[:,5]
        # self.exptriangle = self.triangles[:,6]
        # self.angs = self.triangles[:,7]
        self.get_metrics()
        # self.danger_level = rf.predict(self.metrics_grid)

    def home_plate(self):
        y_upper = np.where(self.xgrid <= 31, 35.05+0.95*self.xgrid, 64.5)
        y_lower = np.where(self.xgrid <= 31, 49.95-0.95*self.xgrid, 20.5)
        square = (self.xgrid <= 46) * (self.xgrid >= 11) * \
            (self.ygrid <= y_upper) * (self.ygrid >= y_lower)
        return self.rcgrid[square].mean()  # self.grid[square,3].mean()

    # def passer_location(self):
    #     #use player motion, center of circle, to predict where they will be.
    #     #return location value at that point
    #     x_passer = self.c_x[:,self.puck].mean()
    #     y_passer = self.c_y[:,self.puck].mean()
    #     return self.locvalgrid[((x_passer-self.xgrid)**2+(y_passer-self.ygrid)**2).argmin()]

    # def metrics_offense(self):
    #     #use player motion, center of circle, to predict where they will be.
    #     #delete player with puck when needed
    #     x_center = self.c_x.mean(axis=0)
    #     y_center = self.c_y.mean(axis=0)
    #     #find angle between passer and each player
    #     #adjust angles to be within -pi and pi
    #     player_angs = np.pi/2+np.arctan2(np.delete(y_center,self.puck)-y_center[self.puck],np.delete(x_center,self.puck)-x_center[self.puck])
    #     player_angs[player_angs>np.pi]-=2*np.pi
    #     #find passing angles which are closest to each players angle and get metrics vals there
    #     vals_at_players = [self.triangles[np.abs(player-self.angs).argmin(),4:7] for player in player_angs]
    #     #return the x,y,success,best,exp for each player excluding the passer
    #     return (np.delete(x_center,self.puck),np.delete(y_center,self.puck),vals_at_players)

    def get_metrics(self):
        y_2_goal = self.yp-GLY
        x_2_goal = self.xp-GLX
        self.metrics_grid = np.array([((x_2_goal)**2+(y_2_goal)**2)**0.5,
                                      self.rcgrid.mean(),
                                      self.ind_var_calculation(),
                                      self.home_plate(),
                                      np.arctan2(
                                          y_2_goal, x_2_goal) * 180 / np.pi,
                                      self.off.sum()+1
                                      ]).reshape(1, -1)
        # self.triangles.max(axis=0)[4:],
        # self.passer_location(),
        # self.metrics_offense(),
        # look within self.metrics_offense() to find mean/max and which player has those if we want

    def mst_properties(self, player_positions, player_teams=None):
        no_players = len(player_positions)  # number of players
        # MST calculation
        # p2p_distances = np.empty(shape=(no_players, no_players))
        # coordinates = player_positions.T
        p2p_distances = distance_matrix(player_positions, player_positions)
        # for i in range(no_players):
        #     p2p_distances[i] = np.sqrt((coordinates[0]-player_positions[i][0])**2 + (coordinates[1]-player_positions[i][1])**2)
        tree = minimum_spanning_tree(csr_matrix(p2p_distances)).toarray()

        # avg edge length and total edge length
        avg_edge_length = np.mean(tree[~(tree == 0)])
        tot_edge_length = np.sum(tree[~(tree == 0)])
        plotted = np.argwhere(tree > 0)

        # counting the number of edges each player has and taking the average of that array ot get avg_edges_per_player
        # edge_player_connections = []
        # for i in plotted:
        #     j,k = i
        #     edge_player_connections.append(j)
        #     edge_player_connections.append(k)
        unique_vals, edge_player_connections = np.unique(
            plotted, return_counts=True)
        avg_edges_per_player = np.mean(edge_player_connections)
        # opponent connection ratio calculation: assign each MST edge a 0 (teamate to teamate connection) or a 1 (opponent to opponent connection) and take the mean of the assigned MST values
        # closer to 1 means more people are paired up with opponents and closer to 0 means paired up with teamates
        if player_teams is not None:
            team_1 = player_teams[plotted[:, 0]]
            team_2 = player_teams[plotted[:, 1]]
            opponent_connection_ratio = np.mean(team_1 != team_2)
            return avg_edge_length, avg_edges_per_player, opponent_connection_ratio
        else:
            return avg_edge_length, avg_edges_per_player

    def ind_var_calculation(self):
        # MST variable calculations
        # MST variable calculations
        x_coords = self.x
        y_coords = self.y
        # for 5 on 4 events  where we have tracking data proceed with MST calculations

        offense = np.array(self.off)
        off_team_strength = len(offense[offense == 1])
        def_team_strength = len(offense[offense == -1])
        # getting coordinates, positions, and respective teams (last one if for OCR calculation) for event
        raw_coord_pairs = np.array([x_coords, y_coords]).T

        # MST calculation
        # getting rid of goalies for calculation of all player MST
        all_coord_pairs = np.delete(raw_coord_pairs, goalie, axis=0)
        player_teams = np.delete(offense, goalie, axis=0)
        # variable calculations for MST properties with all players
        all_avg_edge, all_avg_edges_per_player, all_ocr = self.mst_properties(
            all_coord_pairs, player_teams)

        #off_no_goalie = np.delete(offense, goalie,axis=0)
        # excluding goalie and empty coordinate spots
        #home_coord_pairs = all_coord_pairs[np.where(off_no_goalie==1)]
        #off_avg_edge, off_avg_edges_per_player = self.mst_properties(home_coord_pairs)

        # leaving goalie in for defensive team because it matters to the model
        #away_coord_pairs = raw_coord_pairs[np.where(offense==-1)]
        #def_avg_edge, def_avg_edges_per_player = self.mst_properties(away_coord_pairs)

        # calculating MST ratio betweeen offensive and defense average edge length
        #od_MST_ratio = off_avg_edge/def_avg_edge
        return all_ocr  # ,all_avg_edge, all_avg_edges_per_player,  off_avg_edge, off_avg_edges_per_player, def_avg_edge, def_avg_edges_per_player


if __name__ == '__main__':
    x = list(200 - np.array([171.4262, 155.6585, 153.7146,
             150.5869, 156.3463, 179.8383, 180.8131, 186.6146, 179.9982]))
    y = list(np.array([49.31514, 48.25991, 70.17542, 13.65429,
             28.51970, 38.44596, 36.80571, 38.32781, 22.03946]))
    vx = list(np.array([6.725073,  4.964445, -3.097599, 14.252625,
              4.286796,  1.925091, -2.295729, -0.294258,  6.464229]))
    vy = list(np.array([-7.1037417,  -7.9677960,  -6.4446342,   6.5618985, -
              10.9455216,  -4.7444208,  -4.1465373,  -0.3377985, -5.4265284]))
    goalie = 7  # zero-indexed
    puck = 3
    off = list(np.array([-1, -1, 1, 1, -1, 1, -1, -1, 1]))
    # all_tracks = tracks(x,y,vx,vy,goalie,puck,off) #Robyn - changed tracks to metrics
    # print(all_tracks.triangles.shape)

    # final_metrics = pd.DataFrame(columns=['distance_to_net', 'Rink_Control', 'All_OCR', 'Home_Plate_Control', 'angle_to_attacking_net', 'woman_adv'])
    # final_metrics.loc[0] = all_tracks.metrics_grid
    # with open('finalized_rf_model.pkl','rb') as f:
    #     loaded_rf = pickle.load(f)
    # joblib.dump(loaded_rf,'loaded_rf.joblib')

    # loaded_pred = loaded_rf_2.predict(all_tracks.metrics_grid)
    # print(final_metrics)
    # print(loaded_pred)


# Robyn - load the model from git
