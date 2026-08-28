std_msgs/Header header      # same stamp as vo/pose
string   tracking_state     # stateName(state)
uint32   vo_epoch           # ++ on MapChanged() or on leaving OK
uint32   n_map_points       # -> g_feat
uint32   n_keypoints        # diagnostic
bool     pose_valid         # state == OK or OK_KLT