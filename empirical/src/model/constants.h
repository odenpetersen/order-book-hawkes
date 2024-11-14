#ifndef CONSTANTS_H
#define CONSTANTS_H

#define TRADES_ONLY 1
#define NUM_EVENT_TYPES 4

//#define TRADES_ONLY 0
//#define NUM_EVENT_TYPES 16

//#define START_TIME 0.0l
//#define END_TIME 86400.0l

#define START_TIME 36000.0l
#define END_TIME 37200.0l

#define NUM_MARK_VARIABLES 26
#define LR_START 1e-9//1e-4
#define LR_DECAY 1e-3
#define LR_END 1.0
#define NOISE_START 1.0l
#define NOISE_DECAY 1e-4
#define ATOL 1e-1//1e-5
#define RTOL 1e-1//1e-2

#endif //CONSTANTS_H
