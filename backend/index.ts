import { router,json } from '@appdeploy/sdk';
export const handler=router({'GET /api/_healthcheck':[async()=>json({message:'Success'})]});
