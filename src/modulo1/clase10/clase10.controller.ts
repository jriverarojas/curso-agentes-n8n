import { Body, Controller, Get, Param, Post, UseGuards } from '@nestjs/common';
import { ApiKeyGuard } from '../../auth/guards/api-key.guard';
import { Clase10Service } from './clase10.service';

@Controller('modulo1/clase10')
@UseGuards(ApiKeyGuard)
export class Clase10Controller {
  constructor(private readonly clase10: Clase10Service) {}
  @Get('applications')
  @UseGuards(ApiKeyGuard)
  async listApplications() {
    return await this.clase10.listApplications();
  }

  @Get('document-types')
  @UseGuards(ApiKeyGuard)
  async listDocumentTypes() {
    return await this.clase10.listDocumentTypes();
  }


  @Post('applications/:applicationId/documents')
  @UseGuards(ApiKeyGuard)
  async uploadDocument(
    @Param('applicationId') applicationId: string,
    @Body()
    body: {
      documentType: string;
      fileName: string;
      contentType?: string;
      contentBase64: string;
    },
  ) {
    return await this.clase10.uploadDocument(applicationId, body);
  }

  @Get('applications/:applicationId')
  @UseGuards(ApiKeyGuard)
  async getApplication(@Param('applicationId') applicationId: string) {
    return await this.clase10.getApplication(applicationId);
  }

  @Post('applications')
  @UseGuards(ApiKeyGuard)
  async createApplication(@Body() body: { applicantExternalId?: string; applicantName?: string }) {
    return await this.clase10.createApplication(body);
  }
  
}