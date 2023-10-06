OP_EPS = 0.10


def check_desired_op(closest_op, desired_op):
    return abs((closest_op - desired_op) / desired_op) < OP_EPS
