#include "KEY.h"
#include "stdio.h"
#include "usart.h"
void Key_Init(void)
{
	RCC_APB2PeriphClockCmd(RCC_APB2Periph_GPIOB,ENABLE);
	GPIO_InitTypeDef GPIO_InitStructure;
	GPIO_InitStructure.GPIO_Mode = GPIO_Mode_IPU;
	GPIO_InitStructure.GPIO_Pin = GPIO_Pin_4 | GPIO_Pin_5 | GPIO_Pin_8 | GPIO_Pin_9;
	GPIO_InitStructure.GPIO_Speed = GPIO_Speed_50MHz;
	GPIO_Init(GPIOB,&GPIO_InitStructure);
//	GPIO_PinRemapConfig(GPIO_Remap_SWJ_NoJTRST,ENABLE);//解除 PB4 的 JTAG 占用
}

void KeyTest(void)
{
	
	if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_4)==RESET){
		Delay_ms(20);
		if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_4)==RESET){
			LED_ON();
		}
	}
	else if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_5)==RESET){
		Delay_ms(20);
		if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_5)==RESET){
			LED_OFF();
		}
	}	
	else if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_8)==RESET){
		Delay_ms(20);
		if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_8)==RESET){
			//printf("p0.pic=0\xff\xff\xff");
			printf("n0.val=%d\xff\xff\xff",Juli);
		}
	}
	else if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_9)==RESET){
		Delay_ms(20);
		if(GPIO_ReadInputDataBit(GPIOB,GPIO_Pin_9)==RESET){
			//printf("p0.pic=1\xff\xff\xff");
			printf("n0.val=%d\xff\xff\xff",Juli);
		}
	}	
}

