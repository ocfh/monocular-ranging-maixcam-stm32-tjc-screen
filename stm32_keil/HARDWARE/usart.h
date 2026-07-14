#ifndef __USARY_H
#define __USART_H

extern uint8_t Serial_TxPacket[];
extern char Serial_RxPacket[];
extern uint8_t Serial_RxFlag;
extern uint8_t Juli;
void Serial_SendByte(uint8_t Byte);
void Serial_Init(void);
void Serial2_Init(void);
void Serial_SendArray(uint8_t *Array , uint16_t Length);
void Serial_SendString(char *String);
void Serial_SendNumber(uint32_t Number, uint8_t Length);
void Serial_Printf(char *format, ...);
uint8_t Serial_GetRxData(void);
uint8_t Serial_GetRxFlag(void);
void Serial_SendPacket(void);
#endif
