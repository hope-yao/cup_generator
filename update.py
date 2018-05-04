pg_reinforce = PolicyGradientREINFORCE(sess,
                                       optimizer,
                                       policy_network,
                                       state_dim,
                                       num_actions,
                                       summary_writer=writer)