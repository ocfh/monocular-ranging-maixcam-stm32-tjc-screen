#ifndef __LED_H
#define __LED_H
	
#include "stm32f10x.h"                  // Device header

#define LED_ON()  GPIO_SetBits(GPIOC,GPIO_Pin_13)
#define LED_OFF()  GPIO_ResetBits(GPIOC,GPIO_Pin_13)


void LED_Init(void);


#endif

