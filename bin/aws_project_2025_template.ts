#!/usr/bin/env node
import * as cdk from 'aws-cdk-lib';
import { AwsProject2025TemplateStack } from '../lib/aws_project_2025_template-stack';

const app = new cdk.App();
new AwsProject2025TemplateStack(app, 'AwsProject2025TemplateStack', {
  stackName: 'aws-project-2025-user',
  env: { region: 'ap-northeast-1' },
});