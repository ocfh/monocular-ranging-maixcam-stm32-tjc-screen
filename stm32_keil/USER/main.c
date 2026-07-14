#include "stm32f10x.h"                  // Device header
#include "Delay.h"
#include "LED.h"
#include "string.h"
#include "usart.h"
#include "KEY.h"
#include "stdio.h"

int main(void)
{
	uint8_t ch=0;
	LED_Init();
	NVIC_PriorityGroupConfig(NVIC_PriorityGroup_1);
	Serial_Init();
	Key_Init();
	
	
	while (1)
	{
		Delay_ms(1000);
		//printf("n0.val=%d\xff\xff\xff",ch+=50);
		KeyTest();
		
//		while(USART_GetFlagStatus(USART1,USART_FLAG_RXNE) == RESET );//等待接收结束
//		ch = USART_ReceiveData(USART1);
//		USART_SendData(USART1,ch);
//		while(USART_GetFlagStatus(USART1,USART_FLAG_TXE) == RESET );//等待发送结束
//		
//		LED_ON();
//		Delay_s(1);
//		USART_SendData(USART1, '1');
//		while(!(USART1->SR & (1<<7)));
//		USART_SendData(USART1, '2');
//		while(USART_GetFlagStatus(USART1,USART_FLAG_TXE) == RESET );
//		USART_SendData(USART1, '3');
//		while(!(USART1->SR & (1<<7)));
//		LED_OFF();
//		Delay_s(1);
	}
}


